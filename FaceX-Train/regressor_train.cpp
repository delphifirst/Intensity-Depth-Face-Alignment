/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2015  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#include "regressor_train.h"

#include <utility>
#include <iostream>
#include <memory>
#include <algorithm>

#include "utils_train.h"


using namespace std;

RegressorTrain::RegressorTrain(const TrainingParameters &tp)
	: training_parameters_(tp)
{
	ferns = vector<FernTrain>(training_parameters_.K, FernTrain(tp));
	pixels = std::vector<Pixel>(training_parameters_.P);
}

void RegressorTrain::Regress(const vector<cv::Point2d> &mean_shape,
	vector<vector<cv::Point2d>> *targets,
	const vector<DataPoint> & training_data)
{
	vector<int> type(training_parameters_.P);
	for (int i = 0; i < training_parameters_.P; ++i)
	{
		pixels[i].type = cv::theRNG().uniform(0.0, 1.0) > training_parameters_.Rho;

		type[i] = pixels[i].type;

		pixels[i].base_landmark = cv::theRNG().uniform(0, 
			training_data[0].landmarks.size());
		pixels[i].offset.x = cv::theRNG().uniform(-training_parameters_.Kappa, 
			training_parameters_.Kappa);
		pixels[i].offset.y = cv::theRNG().uniform(-training_parameters_.Kappa, 
			training_parameters_.Kappa);
	}

	unique_ptr<double[]> pixels_val_data(new double[
		training_parameters_.P * training_data.size() + 3]);

	cv::Mat pixels_val(training_parameters_.P, training_data.size(), CV_64FC1, cv::alignPtr(pixels_val_data.get(), 32));
	for (int i = 0; i < pixels_val.cols; ++i)
	{
		Transform t = Procrustes(training_data[i].init_shape, mean_shape);
		vector<cv::Point2d> offsets(training_parameters_.P);
		for (int j = 0; j < training_parameters_.P; ++j)
			offsets[j] = pixels[j].offset;
		t.Apply(&offsets, false);

		vector<cv::Point> pixels_pos;
		for (int j = 0; j < training_parameters_.P; ++j)
			pixels_pos.push_back(training_data[i].init_shape[pixels[j].base_landmark] + offsets[j]);

		pair<double, double> coeffs = ComputePose(training_data[i].image_depth, pixels_pos);
		
		for (int j = 0; j < training_parameters_.P; ++j)
		{
			if (pixels_pos[j].inside(cv::Rect(0, 0,
				training_data[i].image_infrared.cols, training_data[i].image_infrared.rows)))
			{
				if (pixels[j].type == 0)
				{
					pixels_val.at<double>(j, i) =
						training_data[i].image_infrared.at<ushort>(pixels_pos[j]);
				}
				else
				{
					pixels_val.at<double>(j, i) = training_data[i].image_depth.at<ushort>(pixels_pos[j])
						- training_parameters_.Alpha * (coeffs.first * pixels_pos[j].x + coeffs.second * pixels_pos[j].y);
				}
			}
			else
				pixels_val.at<double>(j, i) = 0;
		}
	}

	cv::Mat pixels_cov, means;
	cv::calcCovarMatrix(pixels_val, pixels_cov, means,
		cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS);

	for (int i = 0; i < training_parameters_.K; ++i)
	{
		ferns[i].Regress(targets, type, pixels_val, pixels_cov);
		for (int j = 0; j < targets->size(); ++j)
		{
			(*targets)[j] = ShapeDifference((*targets)[j], ferns[i].Apply(
				pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
		}
	}

	CompressFerns();
}

void RegressorTrain::CompressFerns()
{
	base.create(ferns[0].outputs[0].size() * 2, training_parameters_.Base, CV_64FC1);
	vector<int> rand_index;
	for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
		rand_index.push_back(i);
	random_shuffle(rand_index.begin(), rand_index.end());
	for (int i = 0; i < training_parameters_.Base; ++i)
	{
		const vector<cv::Point2d> &output = ferns[rand_index[i] >> training_parameters_.F]
			.outputs[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		for (int j = 0; j < output.size(); ++j)
		{
			base.at<double>(j * 2, i) = output[j].x;
			base.at<double>(j * 2 + 1, i) = output[j].y;
		}
		cv::normalize(base.col(i), base.col(i));
	}

	for (int i = 0; i < training_parameters_.K; ++i)
	{
		for (int j = 0; j < (1 << training_parameters_.F); ++j)
		{
			const vector<cv::Point2d> &output = ferns[i].outputs[j];
			cv::Mat output_mat(base.rows, 1, CV_64FC1);
			for (int k = 0; k < output.size(); ++k)
			{
				output_mat.at<double>(k * 2) = output[k].x;
				output_mat.at<double>(k * 2 + 1) = output[k].y;
			}

			ferns[i].outputs_mini.push_back(OMP(output_mat, base, training_parameters_.Q));
		}
	}
}

vector<cv::Point2d> RegressorTrain::Apply(const vector<cv::Point2d> &mean_shape, 
	const DataPoint &data) const
{
	cv::Mat pixels_val(1, training_parameters_.P, CV_64FC1);
	Transform t = Procrustes(data.init_shape, mean_shape);
	vector<cv::Point2d> offsets(training_parameters_.P);
	for (int j = 0; j < training_parameters_.P; ++j)
		offsets[j] = pixels[j].offset;
	t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < training_parameters_.P; ++j)
	{
		cv::Point pixel_pos = data.init_shape[pixels[j].base_landmark] + offsets[j];
		if (pixel_pos.inside(cv::Rect(0, 0, data.image_infrared.cols, data.image_infrared.rows)))
		{
			if (pixels[j].type == 0)
				p[j] = data.image_infrared.at<ushort>(pixel_pos);
			else
				p[j] = data.image_depth.at<ushort>(pixel_pos);
		}
		else
			p[j] = 0;
	}

	vector<double> coeffs(training_parameters_.Base);
	for (int i = 0; i < training_parameters_.K; ++i)
		ferns[i].ApplyMini(pixels_val, coeffs);

	cv::Mat result_mat = cv::Mat::zeros(mean_shape.size() * 2, 1, CV_64FC1);
	for (int i = 0; i < training_parameters_.Base; ++i)
		result_mat += coeffs[i] * base.col(i);
	vector<cv::Point2d> result(mean_shape.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}
	return result;
}


void RegressorTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	fs << "pixels";
	fs << "[";
	for (auto it = pixels.begin(); it != pixels.end(); ++it)
		fs << "{" << "type" << it->type << "base_landmark" << it->base_landmark << "offset" << it->offset << "}";
	fs << "]";
	fs << "ferns" << "[";
	for (auto it = ferns.begin(); it != ferns.end(); ++it)
		fs << *it;
	fs << "]";
	fs << "base" << base;
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r)
{
	r.write(fs);
}