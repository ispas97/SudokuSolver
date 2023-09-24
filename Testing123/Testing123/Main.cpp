#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include "FindingLinesFunctions.h"
#include "AdditionalFunctions.h"
#include "FindingHomographyFunctions.h"

# define pi         3.14159265358979323846

using namespace cv;
using namespace std;




int main(int argc, char** argv)
{

	int rows = 628;
	int cols = 1200;

	String s = "sudoku5.jpg";
	Mat_<unsigned __int8> image = imread(s, IMREAD_GRAYSCALE);
	Mat_<unsigned __int8> imageCanny;
	Mat_<unsigned __int8> outputImage;
	Canny(image, imageCanny, 175, 200);
	waitKey(0);
	unsigned __int8* picture;
	picture = new unsigned __int8[rows * cols];

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			picture[cols * i + j] = imageCanny[i][j];
		}
	imshow("Original Image", image);
	std::vector<int> outputArrayWithoutFiltering = doPicture(picture, cols, rows);
	DrawLines(imageCanny, outputArrayWithoutFiltering);
	cv::imshow("CannyBeforeFiltering", imageCanny);
	std::vector<int> outputArrayWithFiltering = Filtering(outputArrayWithoutFiltering, 20, 5);
	std::vector<int> arrayVer;
	std::vector<int> arrayHor;
	for (std::vector<int>::iterator it = outputArrayWithFiltering.begin(); it != outputArrayWithFiltering.end();)
	{
		if ((*(it + 1) > 8) && (*(it + 1) < 13))//(12,25)
		{
			arrayVer.push_back(*it);
			arrayVer.push_back(*(it + 1));
		}
		else if ((*(it + 1) > 280) && (*(it + 1) < 283))
		{
			arrayHor.push_back(*it);
			arrayHor.push_back(*(it + 1));
		}
		it += 2;
	}
	std::vector<int> arrayHorFiltered = Filtering(arrayHor, 20, 5);
	std::vector<int> arrayVerFiltered = Filtering(arrayVer, 20, 5);
	DrawLines(imageCanny, arrayHorFiltered);
	DrawLines(imageCanny, arrayVerFiltered);
	imshow("Output2", imageCanny);
	waitKey(0);
	Mat H(3, 3, CV_32F);
	H = FindingHomographyFunc(image, arrayHorFiltered, arrayVerFiltered, 55);
	Size sz(1300, 900);
	cv::warpPerspective(image, outputImage, H, sz);
	imshow("Output1", outputImage);



	waitKey(0);
	return 0;
}