#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include "AdditionalFunctions.h"


# define pi         3.14159265358979323846

using namespace cv;
using namespace std;

cv::Mat CalculateHomography(std::vector<Point2f> a, std::vector<Point2f> b)
{
	Mat P(9, 9, CV_32F, Scalar(0));
	for (int i = 0; i < 4; i++)
	{
		float x1_a = a[i].x;
		float x1_b = b[i].x;
		float y1_a = a[i].y;
		float y1_b = b[i].y;
		P.at<float>(2 * i, 0) = -x1_a;
		P.at<float>(2 * i, 1) = -y1_a;
		P.at<float>(2 * i, 2) = -1;
		P.at<float>(2 * i, 6) = x1_a * x1_b;
		P.at<float>(2 * i, 7) = y1_a * x1_b;
		P.at<float>(2 * i, 8) = x1_b;
		P.at<float>(2 * i + 1, 3) = -x1_a;
		P.at<float>(2 * i + 1, 4) = -y1_a;
		P.at<float>(2 * i + 1, 5) = -1;
		P.at<float>(2 * i + 1, 6) = x1_a * y1_b;
		P.at<float>(2 * i + 1, 7) = y1_a * y1_b;
		P.at<float>(2 * i + 1, 8) = y1_b;
	}
	P.at<float>(8, 8) = 1;
	Mat P_inv(9, 9, CV_32F, Scalar(0));
		P_inv = P.inv();

	Mat row_mat(9, 1, CV_32F, Scalar(0));
	row_mat.at<float>(8, 0) = 1;
	Mat H = P_inv * row_mat;

	Mat H3X3(3, 3, CV_32F, Scalar(0));
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
		{
			H3X3.at<float>(row, col) = H.at<float>(3 * row + col, 0);
		}
	return H3X3;

}
Point2f CalculatePoint(int rho_1, int theta_1, int rho_2, int theta_2)
{
	float x = (sin((float)(theta_2)*pi / 180) * (float)(rho_1)-sin((float)(theta_1)*pi / 180) * (float)(rho_2)) / (cos((float)(theta_1)*pi / 180) * sin((float)(theta_2)*pi / 180) - cos((float)(theta_2)*pi / 180) * sin((float)(theta_1)*pi / 180));
	float y = (cos((float)(theta_1)*pi / 180) * (float)(rho_2)-cos((float)(theta_2)*pi / 180) * (float)(rho_1)) / (sin((float)(theta_1)*pi / 180) * cos((float)(theta_2)*pi / 180) - sin((float)(theta_2)*pi / 180) * cos((float)(theta_1)*pi / 180));
	Point2f p(x, y);
	return p;
}
void sort(std::vector<int> array)
{
	for (int i = 0; i < array.size() - 2; i += 2)
		for (int j = i + 2; j < array.size(); j += 2)
			if (array[i] > array[j])
			{
				int t1 = array[i];
				int t2 = array[i + 1];
				array[i] = array[j];
				array[i + 1] = array[j + 1];
				array[j] = t1;
				array[j + 1] = t2;
			}
}
std::vector<Point2f> CalculateIntersections(std::vector<int> array_1, std::vector<int> array_2)
{
	sort(array_1);
	sort(array_2);
	std::vector<Point2f> vec;
	for (int i = 0; i < array_1.size(); i += 2)
		for (int j = 0; j < array_2.size(); j += 2)
		{
			Point2f p = CalculatePoint(array_1[i], array_1[i + 1], array_2[j], array_2[j + 1]);
			vec.push_back(p);
		}
	return vec;
}
float distance(Point2f p1, Point2f p2)
{
	return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}
void swap(int* x, int* y)
{
	int t = *x;
	*x = *y;
	*y = t;
}
void DrawLines(Mat image, std::vector<int> array1)
{
	for (int i = 0; i < array1.size(); i += 2)
	{
		float a = 1 / tan((float)array1[i + 1] * pi / 180);
		float b = -(float)array1[i] / sin((float)array1[i + 1] * pi / 180);
		int x0 = -2000;
		int y0 = (int)(x0 * a + b);
		int x1 = 2000;
		int y1 = (int)(x1 * a + b);
		Point t1(x0, y0);
		Point t2(x1, y1);
		cv::line(image, t1, t2, Scalar(255), 3, 8, 0);
	}
}
void DrawLines2(Mat image, std::vector<int> array)
{
	for (int i = 0; i < array.size(); i += 2)
	{
		float rho = array[i];
		float theta = array[i + 1];
		double a = cos(theta * pi / 180);
		double b = sin(theta * pi / 180);
		double x0 = a * rho;
		double y0 = b * rho;
		cv::Point p1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		cv::Point p2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
		cv::line(image, p1, p2, Scalar(255), 3, 8, 0);
	}
}
Mat to_Mat(Point2f p)
{
	Mat P(3, 1, CV_32F);
	P.at<float>(0, 0) = p.x;
	P.at<float>(1, 0) = p.y;
	P.at<float>(2, 0) = 1;
	return P;
}