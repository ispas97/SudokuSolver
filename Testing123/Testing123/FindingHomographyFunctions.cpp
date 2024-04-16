#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include "FindingHomographyFunctions.h"
#include "AdditionalFunctions.h"



# define pi         3.14159265358979323846

using namespace cv;
using namespace std;




int BelongsToGrid(Mat X_warped,Mat P1_warped,float grid_square_width,float tolerance,float belongs_to_picture_tolerance)
{
	if ((X_warped.at<float>(0, 0) > (9 * grid_square_width + belongs_to_picture_tolerance)) && (X_warped.at<float>(1, 0) > (9 * grid_square_width + belongs_to_picture_tolerance)))
		return -1;
	if ((X_warped.at<float>(0, 0) > -belongs_to_picture_tolerance) && (X_warped.at<float>(1, 0) > -belongs_to_picture_tolerance))
	{
		float p1_to_p_x_distance = X_warped.at<float>(0, 0) - P1_warped.at<float>(0, 0);
		float p1_to_p_y_distance = X_warped.at<float>(1, 0) - P1_warped.at<float>(1, 0);
		float k1 = p1_to_p_x_distance / grid_square_width;
		float k1_round = round(k1);
		float k2 = p1_to_p_y_distance / grid_square_width;
		float k2_round = round(k2);
		if ((abs(k1 - k1_round) < tolerance) && (abs(k2 - k2_round) < tolerance))
			return 1;
		return 0;
	}
	return -1;
}
int CheckMissingVerticalEdges(Mat image_cannied, float grid_square_width, float left_most_x, float right_most_x, int tolerance)
{
	float leftEdge = left_most_x - grid_square_width;
	float rightEdge = right_most_x + grid_square_width;
	int leftVotes = 0;
	int rightVotes = 0;
	for (int i = leftEdge - tolerance; i < leftEdge + tolerance; i++)
		for (int j = 0; j < image_cannied.rows; j++)
			if (image_cannied.at<unsigned __int8>(j, i) == 255)
				leftVotes++;
	for (int i = rightEdge - tolerance; i < rightEdge + tolerance; i++)
		for (int j = 0; j < image_cannied.rows; j++)
			if (image_cannied.at<unsigned __int8>(j, i) == 255)
				rightVotes++;
	if (rightVotes > leftVotes)
		return 1;
	else return -1;
}
int CheckMissingHorizontalEdges(Mat image_cannied, float grid_square_width, float upper_most_y, float lower_most_y, int tolerance)
{
	float upEdge = upper_most_y - grid_square_width;
	float downEdge = lower_most_y + grid_square_width;
	int upVotes = 0;
	int downVotes = 0;
	for (int i = upEdge - tolerance; i < upEdge + tolerance; i++)
		for (int j = 0; j < image_cannied.cols; j++)
			if (image_cannied.at<unsigned __int8>(i, j) == 255)
				upVotes++;
	for (int i = downEdge - tolerance; i < downEdge + tolerance; i++)
		for (int j = 0; j < image_cannied.cols; j++)
			if (image_cannied.at<unsigned __int8>(i, j) == 255)
				downVotes++;
	if (downVotes > upVotes)
		return 1;
	else return -1;
}
std::vector<Point2f> CornerPoints(vector<int> array_hor, vector<int> array_ver)
{
	int line1_index = 0;
	int line2_index = array_hor.size() - 2;
	int line3_index = 0;
	int line4_index = array_ver.size() - 2;

	int line1_rho = array_hor[line1_index];
	int line1_theta = array_hor[line1_index + 1];
	int line2_rho = array_hor[line2_index];
	int line2_theta = array_hor[line2_index + 1];
	int line3_rho = array_ver[line3_index];
	int line3_theta = array_ver[line3_index + 1];
	int line4_rho = array_ver[line4_index];
	int line4_theta = array_ver[line4_index + 1];

	Point2f p1 = CalculatePoint(line1_rho, line1_theta, line3_rho, line3_theta);
	Point2f p2 = CalculatePoint(line1_rho, line1_theta, line4_rho, line4_theta);
	Point2f p3 = CalculatePoint(line2_rho, line2_theta, line4_rho, line4_theta);
	Point2f p4 = CalculatePoint(line2_rho, line2_theta, line3_rho, line3_theta);

	std::vector<Point2f> point_vec;
	point_vec.push_back(p1);
	point_vec.push_back(p2);
	point_vec.push_back(p3);
	point_vec.push_back(p4);
	return point_vec;

}
Mat MoveBoard(std::vector<Point2f> src_array, Mat H, float grid_square_width,int* grid_pos_array,int margin)
{
	Mat P1 = to_Mat(src_array[0]);
	Mat P2 = to_Mat(src_array[1]);
	Mat P3 = to_Mat(src_array[2]);
	Mat P1_warped = H * P1;
	Mat P2_warped = H * P2;
	Mat P3_warped = H * P3;

	float left_most_x = P1_warped.at<float>(0, 0);
	float right_most_x = P2_warped.at<float>(0, 0);
	float furthest_x_distance = right_most_x - left_most_x;

	float upper_most_y = P2_warped.at<float>(1, 0);
	float lower_most_y = P3_warped.at<float>(1, 0);
	float furthest_y_distance = lower_most_y - upper_most_y;

	float k = lower_most_y / grid_square_width;
	int lowest_y = round(k);
	int going_down = 9 - lowest_y;

	k = right_most_x / grid_square_width;
	int rightest_x = round(k);
	int going_right = 9 - rightest_x;

	Point2f p5(grid_square_width * grid_pos_array[2] + going_right * grid_square_width+margin, grid_square_width * grid_pos_array[3] + going_down * grid_square_width + margin);
	Point2f p6(grid_square_width * grid_pos_array[2] + furthest_x_distance + going_right * grid_square_width + margin, grid_square_width * grid_pos_array[3] + going_down * grid_square_width + margin);
	Point2f p7(grid_square_width * grid_pos_array[2] + furthest_x_distance + going_right * grid_square_width + margin, grid_square_width * grid_pos_array[3] + furthest_y_distance + going_down * grid_square_width + margin);
	Point2f p8(grid_square_width * grid_pos_array[2] + going_right * grid_square_width + margin, grid_square_width * grid_pos_array[3] + furthest_y_distance + going_down * grid_square_width + margin);

	std::vector<Point2f> dst_array;
	dst_array.push_back(p5);
	dst_array.push_back(p6);
	dst_array.push_back(p7);
	dst_array.push_back(p8);
	Mat homography = CalculateHomography(src_array, dst_array);
	return homography;
}
std::vector<float> Lowest_X_Biggest_X (Mat P1, Mat P2, Mat image_cannied, float grid_square_width, std::vector<float> edge_array)
{
	float left_most_x = P1.at<float>(0, 0);
	float right_most_x = P2.at<float>(0, 0);
	float furthest_x_distance = right_most_x - left_most_x;
	int num_of_ver_lines_minus_one = round(furthest_x_distance / grid_square_width);

	while (num_of_ver_lines_minus_one < 9)
	{
		int direction = CheckMissingVerticalEdges(image_cannied, grid_square_width, left_most_x, right_most_x);
		if (direction == -1)
			left_most_x = left_most_x - grid_square_width;
		else right_most_x = right_most_x + grid_square_width;
		num_of_ver_lines_minus_one++;
	}
	edge_array.push_back(left_most_x);
	edge_array.push_back(right_most_x);

	return edge_array;
}
std::vector<float> Lowest_Y_Biggest_Y(Mat P1, Mat P2, Mat image_cannied, float grid_square_width, std::vector<float> edge_array)
{
	float upper_most_y = P1.at<float>(1, 0);
	float lower_most_y = P2.at<float>(1, 0);
	float furthest_y_distance = lower_most_y - upper_most_y;
	int num_of_hor_lines_minus_one = round(furthest_y_distance / grid_square_width);

	while (num_of_hor_lines_minus_one < 9)
	{
		int direction = CheckMissingHorizontalEdges(image_cannied,grid_square_width, upper_most_y, lower_most_y);
		if (direction == -1)
			upper_most_y = upper_most_y - grid_square_width;
		else lower_most_y = lower_most_y + grid_square_width;
		num_of_hor_lines_minus_one++;
	}
	edge_array.push_back(upper_most_y);
	edge_array.push_back(lower_most_y);
	return edge_array;
}
std::vector<Point2f> FindSudokuBoardCorners(Mat warped_image, Mat moved_homography, std::vector<Point2f> corner_points, float grid_square_width)
{
	Mat P1 = to_Mat(corner_points[0]);
	Mat P2 = to_Mat(corner_points[1]);
	Mat P3 = to_Mat(corner_points[2]);
	Mat new_P1_warped = moved_homography * P1;
	Mat new_P2_warped = moved_homography * P2;
	Mat new_P3_warped = moved_homography * P3;

	Mat image_sobeled;
	Mat image_cannied;
	cv::Sobel(warped_image, image_sobeled, CV_8UC1, 1, 0);
	cv::Canny(image_sobeled, image_cannied, 175, 200);
	cv::imshow("Image Cannied", image_cannied);
	std::vector<float> edge_array;
	edge_array=Lowest_X_Biggest_X(new_P1_warped, new_P2_warped, image_cannied, grid_square_width, edge_array);
	edge_array=Lowest_Y_Biggest_Y(new_P2_warped, new_P3_warped, image_cannied, grid_square_width, edge_array);
	
	std::vector<Point2f> corners;
	corners.push_back(Point2f(edge_array[0], edge_array[2]));
	corners.push_back(Point2f(edge_array[1], edge_array[2]));
	corners.push_back(Point2f(edge_array[1], edge_array[3]));
	corners.push_back(Point2f(edge_array[0], edge_array[3]));


	return corners;
}
Mat AdjustingHomography(Mat image, Mat H, vector<int> array_hor, vector<int> array_ver, float grid_square_width,int scale,int* grid_pos_array)
{

	std::vector<Point2f> corner_points=CornerPoints(array_hor, array_ver);

	Mat moved_homography=MoveBoard(corner_points, H, grid_square_width, grid_pos_array);

	Mat warped_image;
	Size sz(18*grid_square_width, 18*grid_square_width);
	cv::warpPerspective(image, warped_image, moved_homography, sz);

	std::vector<Point2f> sudoku_corners=FindSudokuBoardCorners(warped_image, moved_homography, corner_points, grid_square_width);
	//MoveBackBoard
	return moved_homography;

}
Mat FindingHomographyFunc(Mat image,std::vector<int> array_hor,std::vector<int> array_ver,int scale, float tolerance)
{
	std::vector<Point2f> intersection_points_vector = CalculateIntersections(array_hor, array_ver);
	int max_num_of_inliers = 0;
	Mat result_homography(3, 3, CV_32F);
	float grid_square_width = scale;
	bool NOT_FOUND = true;
	int grid_pos_array[4];

		while (NOT_FOUND)
		{
			int line1_index = rand() % ((array_hor.size()/2)-1);
			int line2_index = rand() % ((array_hor.size()/2)-1);
			int line3_index = rand() % ((array_ver.size()/2)-1);
			int line4_index = rand() % ((array_ver.size()/2)-1);

			if (line1_index == line2_index)
				line2_index++;
			if (line3_index == line4_index)
				line4_index++;
			if (line1_index > line2_index)
				swap(line1_index, line2_index);
			if (line3_index > line4_index)
				swap(line3_index, line4_index);
			int line1_rho = array_hor[2 * line1_index];
			int line1_theta = array_hor[2 * line1_index + 1];
			int line2_rho = array_hor[2 * line2_index];
			int line2_theta = array_hor[2 * line2_index + 1];
			int line3_rho = array_ver[2 * line3_index];
			int line3_theta = array_ver[2 * line3_index + 1];
			int line4_rho = array_ver[2 * line4_index];
			int line4_theta = array_ver[2 * line4_index + 1];


			Point2f p1 = CalculatePoint(line1_rho, line1_theta, line3_rho, line3_theta);
			Point2f p2 = CalculatePoint(line1_rho, line1_theta, line4_rho, line4_theta);
			Point2f p3 = CalculatePoint(line2_rho, line2_theta, line4_rho, line4_theta);
			Point2f p4 = CalculatePoint(line2_rho, line2_theta, line3_rho, line3_theta);

			for (int sx = 1; sx < 10; sx++)
				for (int sy = 1; sy < 10; sy++)
					for (int swx=0;swx<10-sx;swx+=1)
 						for (int swy=0;swy<10-sy;swy+=1)
							if (NOT_FOUND)
							{
								Point2f p5(scale * swx, scale * swy);
								Point2f p6(scale * swx + scale * sx, scale * swy);
								Point2f p7(scale * swx + scale * sx, scale * swy + scale * sy);
								Point2f p8(scale * swx, scale * swy + scale * sy);
								std::vector<Point2f> src_picture_points;
								std::vector<Point2f> dst_picture_points;

								src_picture_points.push_back(p1);
								src_picture_points.push_back(p2);
								src_picture_points.push_back(p3);
								src_picture_points.push_back(p4);

								dst_picture_points.push_back(p5);
								dst_picture_points.push_back(p6);
								dst_picture_points.push_back(p7);
								dst_picture_points.push_back(p8);

								Mat homography = CalculateHomography(src_picture_points, dst_picture_points);

								Mat P1 = to_Mat(p1);
								Mat P1_warped = homography * P1;

								Mat P2 = to_Mat(p2);
								Mat P2_warped = homography * P2;

								grid_square_width = (P2_warped.at<float>(0, 0) - P1_warped.at<float>(0, 0)) / (float)sx;

								float p2_to_p3_distance = line2_rho - line1_rho;
								float p1_to_p2_distance = line4_rho - line3_rho;

								float ratio_1 = p1_to_p2_distance / p2_to_p3_distance;
								float ratio_2 = (float)sx / (float)sy;
								
								float ratio_difference_tolerance = 0.4;

								if (abs(ratio_1 - ratio_2) < ratio_difference_tolerance)
								{
									int num_of_inliers = 0;

									for (std::vector<Point2f>::iterator it = intersection_points_vector.begin(); it != intersection_points_vector.end(); ++it)
									{
										Mat X(3, 1, CV_32F);
										X.at<float>(0, 0) = (*it).x;
										X.at<float>(1, 0) = (*it).y;
										X.at<float>(2, 0) = 1;
										Mat X_warped = homography * X;

										int a = BelongsToGrid(X_warped, P1_warped, grid_square_width, tolerance);
										if (a == 1)
											num_of_inliers++;
										else if (a == -1)
										{
											num_of_inliers = 0;
											it = intersection_points_vector.end() - 1;
										}
									}
									if (num_of_inliers > max_num_of_inliers)
									{
										bool IS_HOMOGRAPHY_VALID = false;
										for (int row = 0; row < 3; row++)
										{
											for (int col = 0; col < 3; col++)
												if (homography.at<float>(row, col) != 0)
													IS_HOMOGRAPHY_VALID = true;
										}
										if (IS_HOMOGRAPHY_VALID)
										{
											grid_pos_array[0] = sx;
											grid_pos_array[1] = sy;
											grid_pos_array[2] = swx;
											grid_pos_array[3] = swy;
											result_homography = homography;
											max_num_of_inliers = num_of_inliers;
											if (max_num_of_inliers > (array_hor.size() / 2 * array_ver.size() / 2) / 4 + 5)//it has to be greater than hor*ver/4 because of 2x1 and 2x2
													NOT_FOUND = false;
										}
									}
								}
							}			
		}
		Mat warped_image;
		Size sz(18 * grid_square_width, 18 * grid_square_width);
		cv::warpPerspective(image, warped_image, result_homography, sz);
		cv::imshow("Warped Image", warped_image);
		Mat newHomography = AdjustingHomography(image, result_homography, array_hor, array_ver, grid_square_width,scale,grid_pos_array);

	return newHomography;
}
