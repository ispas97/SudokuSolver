#pragma once
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;


	int BelongsToGrid(Mat X_warped, Mat P1_warped, float x_distance, float tolerance, float belongs_to_picture_tolerance = 5);
	int CheckMissingVerticalEdges(Mat imageCannied, float x_distance, float left_most_x, float right_most_x, int tolerance = 5);
	int CheckMissingHorizontalEdges(Mat imageCannied, float x_distance, float upper_most_y, float lower_most_y, int tolerance = 5);
	std::vector<Point2f> IntersectionPoints(vector<int> arrayHor, vector<int> arrayVer);
	Mat MoveBoard(std::vector<Point2f> src_array, Mat H, float x_distance, int* gridPosArray, int margin = 6);
	std::vector<float> Lowest_X_Biggest_X(Mat P1, Mat P2, Mat imageCannied, float x_distance, std::vector<float> edgeArray);
	std::vector<float> Lowest_Y_Biggest_Y(Mat P1, Mat P2, Mat imageCannied, float x_distance, std::vector<float> edgeArray);
	std::vector<Point2f> FindSudokuBoardCorners(Mat warpedImage, Mat movedHomography, std::vector<Point2f> array1, float x_distance);
	Mat AdjustingHomography(Mat image, Mat H, vector<int> arrayHor, vector<int> arrayVer, float x_distance, int scale, int* gridPosArray);
	Mat FindingHomographyFunc(Mat image, std::vector<int> arrayHor, std::vector<int> arrayVer, int scale = 100, float tolerance = 0.05);
