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

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face_x.h"

using namespace std;

template<typename T>
T sqr(T n)
{
	return n * n;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "Usage: FaceX model.xml.gz" << endl;
		return 0;
	}

	FaceX face_x;
	face_x.OpenModel(argv[1]);

	ifstream fin("test\\labels.txt");
	string pathname;

	vector<double> errors;

	while (fin >> pathname)
	{
		int left, right, top, bottom;
		fin >> left >> right >> top >> bottom;
		vector<cv::Point2d> landmarks;
		for (int i = 0; i < face_x.landmarks_count(); ++i)
		{
			cv::Point2d p;
			fin >> p.x >> p.y;
			landmarks.push_back(p);
		}

		cv::Mat image_infrared = cv::imread("test\\" + pathname + "_long_exposure_infrared.png", CV_LOAD_IMAGE_ANYDEPTH);
		cv::Mat image_depth = cv::imread("test\\" + pathname + "_depth.png", CV_LOAD_IMAGE_ANYDEPTH);

		vector<cv::Point2d> landmarks_estimation = face_x.Alignment(
			image_infrared, image_depth, cv::Rect(left, top, right - left, bottom - top));

		double eye_distance = sqrt(sqr(landmarks[5].x - landmarks[1].x) + sqr(landmarks[5].y - landmarks[1].y));

		double error = 0;
		for (int i = 0; i < landmarks.size(); ++i)
		{
			error += sqrt(sqr(landmarks[i].x - landmarks_estimation[i].x) + sqr(landmarks[i].y - landmarks_estimation[i].y));
		}

		errors.push_back(error / face_x.landmarks_count() / eye_distance);

		/*cv::Mat image(image_infrared.rows, image_infrared.cols, CV_8UC3);

		for (int i = 0; i < image.rows; ++i)
			for (int j = 0; j < image.cols; ++j)
			{
				uchar p = image_infrared.at<ushort>(i, j) * 4 / 255;
				image.at<cv::Vec3b>(i, j) = cv::Vec3b(p, p, p);
			}

		for (cv::Point2d p : landmarks_estimation)
		{
			if (p.x >= 0 && p.y >= 0 && p.x < image.cols && p.y < image.rows)
				cv::circle(image, p, 0, cv::Scalar(0, 255, 0), 1);
		}
		cv::imshow("test", image);
		cv::imwrite("result.png", image);
		cv::waitKey();*/
	}

	sort(errors.begin(), errors.end());

	double mean_error = accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
	double median_error = errors[errors.size() / 2];
	double max_error = errors.back();
	double std_error = 0;
	for (double error : errors)
	{
		std_error += sqr(error - mean_error);
	}
	std_error = sqrt(std_error / errors.size());

	cout << "Mean: " << mean_error << endl;
	cout << "Median: " << median_error << endl;
	cout << "Max: " << max_error << endl;
	cout << "Std: " << std_error << endl;
}
