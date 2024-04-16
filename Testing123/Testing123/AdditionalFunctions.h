#pragma once
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

	cv::Mat CalculateHomography(std::vector<Point2f> a, std::vector<Point2f> b);
	Point2f CalculatePoint(int rho_1, int theta_1, int rho_2, int theta_2);
	void sort(std::vector<int> array);
	std::vector<Point2f> CalculateIntersections(std::vector<int> niz1, std::vector<int> niz2);
	float distance(Point2f p1, Point2f p2);
	int CalculateNumOfInliers(std::vector<Point2f> vec, cv::Mat homography, float tolerance);
	void swap(int* x, int* y);
	void DrawLines(Mat image, std::vector<int> array1);
	void DrawLines2(Mat image, std::vector<int> array1);
	Mat to_Mat(Point2f p);
