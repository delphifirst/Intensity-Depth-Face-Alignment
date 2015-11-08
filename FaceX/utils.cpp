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

#include "utils.h"

#include <cassert>
#include <numeric>

using namespace std;

void Transform::Apply(vector<cv::Point2d> *x, bool need_translation)
{
	for (cv::Point2d &p : *x)
	{
		cv::Matx21d v;
		v(0) = p.x;
		v(1) = p.y;
		v = scale_rotation * v;
		if (need_translation)
			v += translation;
		p.x = v(0);
		p.y = v(1);
	}
}

Transform Procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
{
	assert(x.size() == y.size());
	int landmark_count = x.size();
	double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
	double C1 = 0, C2 = 0;

	for (int i = 0; i < landmark_count; ++i)
	{
		X1 += x[i].x;
		X2 += y[i].x;
		Y1 += x[i].y;
		Y2 += y[i].y;
		Z += Sqr(y[i].x) + Sqr(y[i].y);
		C1 += x[i].x * y[i].x + x[i].y * y[i].y;
		C2 += x[i].y * y[i].x - x[i].x * y[i].y;
	}

	cv::Matx44d A(X2, -Y2, W, 0,
		Y2, X2, 0, W,
		Z, 0, X2, Y2,
		0, Z, -Y2, X2);
	cv::Matx41d b(X1, Y1, C1, C2);
	cv::Matx41d solution = A.inv() * b;

	Transform result;
	result.scale_rotation(0, 0) = solution(0);
	result.scale_rotation(0, 1) = -solution(1);
	result.scale_rotation(1, 0) = solution(1);
	result.scale_rotation(1, 1) = solution(0);
	result.translation(0) = solution(2);
	result.translation(1) = solution(3);
	return result;
}

vector<cv::Point2d> ShapeAdjustment(const vector<cv::Point2d> &shape,
	const vector<cv::Point2d> &offset)
{
	assert(shape.size() == offset.size());
	vector<cv::Point2d> result(shape.size());
	for (int i = 0; i < shape.size(); ++i)
		result[i] = shape[i] + offset[i];
	return result;
}

vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const vector<cv::Point2d> original_landmarks, cv::Rect new_face_rect)
{
	vector<cv::Point2d> result;
	for (const cv::Point2d &landmark: original_landmarks)
	{
		result.push_back(landmark);
		result.back() -= cv::Point2d(original_face_rect.x, original_face_rect.y);
		result.back().x *= 
			static_cast<double>(new_face_rect.width) / original_face_rect.width;
		result.back().y *= 
			static_cast<double>(new_face_rect.height) / original_face_rect.height;
		result.back() += cv::Point2d(new_face_rect.x, new_face_rect.y);
	}
	return result;
}

pair<double, double> ComputePose(cv::Mat depth, const vector<cv::Point> & pixels_pos)
{
	vector<cv::Point> valid_pixels_pos;
	for (cv::Point pixel_pos : pixels_pos)
		if (pixel_pos.inside(cv::Rect(0, 0, depth.cols, depth.rows)))
			valid_pixels_pos.push_back(pixel_pos);

	{
		vector<double> all_depth;
		for (cv::Point valid_pixel_pos : valid_pixels_pos)
			all_depth.push_back(depth.at<ushort>(valid_pixel_pos));
		double average = accumulate(all_depth.begin(), all_depth.end(), 0.0) / all_depth.size();
		vector<cv::Point> temp_pixels_pos;
		for (int i = 0; i < valid_pixels_pos.size(); ++i)
			if (abs(average - all_depth[i]) < 100)
				temp_pixels_pos.push_back(valid_pixels_pos[i]);
		valid_pixels_pos = temp_pixels_pos;
	}

	cv::Mat X(valid_pixels_pos.size(), 3, CV_64FC1);
	cv::Mat Y(valid_pixels_pos.size(), 1, CV_64FC1);
	for (int i = 0; i < X.rows; ++i)
	{
		X.at<double>(i, 0) = 1;
		X.at<double>(i, 1) = valid_pixels_pos[i].x;
		X.at<double>(i, 2) = valid_pixels_pos[i].y;
		Y.at<double>(i) = depth.at<ushort>(valid_pixels_pos[i]);
	}

	cv::Mat beta = (X.t() * X).inv() * X.t() * Y;
	pair<double, double> result;
	result.first = beta.at<double>(1);
	result.second = beta.at<double>(2);
	return result;
}