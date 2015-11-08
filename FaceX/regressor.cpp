/*
The MIT License(MIT)

Copyright(c) 2015 Yang Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "regressor.h"

#include <utility>
#include <iostream>
#include <algorithm>

#include "utils.h"


using namespace std;

vector<cv::Point2d> Regressor::Apply(const vector<cv::Point2d> &mean_shape, 
	cv::Mat image_infrared, cv::Mat image_depth, const vector<cv::Point2d> &init_shape) const
{
	cv::Mat pixels_val(1, pixels_.size(), CV_64FC1);
	Transform t = Procrustes(init_shape, mean_shape);
	vector<cv::Point2d> offsets(pixels_.size());
	for (int j = 0; j < pixels_.size(); ++j)
		offsets[j] = pixels_[j].offset;
	t.Apply(&offsets, false);

	vector<cv::Point> pixels_pos;
	for (int j = 0; j < pixels_.size(); ++j)
		pixels_pos.push_back(init_shape[pixels_[j].base_landmark] + offsets[j]);

	pair<double, double> coeffs = ComputePose(image_depth, pixels_pos);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < pixels_.size(); ++j)
	{
		if (pixels_pos[j].inside(cv::Rect(0, 0, image_infrared.cols, image_infrared.rows)))
		{
			if (pixels_[j].type == 0)
				p[j] = image_infrared.at<ushort>(pixels_pos[j]);
			else
			{
				p[j] = image_depth.at<ushort>(pixels_pos[j])
					- alpha_ * (coeffs.first * pixels_pos[j].x + coeffs.second * pixels_pos[j].y);
			}
		}
		else
			p[j] = 0;
	}

	cv::Mat base_coeffs = cv::Mat::zeros(base_.cols, 1, CV_64FC1);
	for (int i = 0; i < ferns_.size(); ++i)
		ferns_[i].ApplyMini(pixels_val, base_coeffs);

	cv::Mat result_mat = base_ * base_coeffs;

	vector<cv::Point2d> result(mean_shape.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}
	return result;
}

void Regressor::read(const cv::FileNode &fn)
{
	pixels_.clear();
	ferns_.clear();
	cv::FileNode pixels_node = fn["pixels"];
	for (auto it = pixels_node.begin(); it != pixels_node.end(); ++it)
	{
		Pixel pixel;
		(*it)["type"] >> pixel.type;
		(*it)["base_landmark"] >> pixel.base_landmark;
		(*it)["offset"] >> pixel.offset;
		pixels_.push_back(pixel);
	}
	cv::FileNode ferns_node = fn["ferns"];
	for (auto it = ferns_node.begin(); it != ferns_node.end(); ++it)
	{
		Fern f;
		*it >> f;
		ferns_.push_back(f);
	}
	fn["base"] >> base_;
}

void read(const cv::FileNode& node, Regressor& r, const Regressor& default_value)
{
	if (node.empty())
	{
		r = default_value;
		cout << "One default Regressor. Model file is corrupt!" << endl;
	}
	else
		r.read(node);
}