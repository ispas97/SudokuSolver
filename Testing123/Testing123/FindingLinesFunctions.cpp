#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include "FindingLinesFunctions.h"

# define pi         3.14159265358979323846

using namespace cv;
using namespace std; 
int* rhoThetaMatrixInitilization(int rho, int theta)
{
	int* rhoThetaMatrix = new int[rho * theta]();
	return rhoThetaMatrix;
}
float CalculatingDRho(int x, int y, float dthetaInRadians)
{
	float drho = (float)x * cos(dthetaInRadians) - (float)y * sin(dthetaInRadians);
	return drho;
}
void IncrementingHoughSpaceMatrixField(int drho, int rho, int* houghSpaceMatrix, int theta, int dtheta)
{
	if ((drho >= 0) && (drho < rho))
		houghSpaceMatrix[theta * drho + dtheta] = houghSpaceMatrix[theta * drho + dtheta] + 1;
}
void GoingThrewAllPossibleValuesOfDTheta2(int x, int y, int rho, int theta, int* houghSpaceMatrix)
{
	for (int dtheta = 0; dtheta < 3600; dtheta++)
	{
		float dthetaInRadians = ((float)dtheta / 10) * pi / 180;
		float drho_1 = CalculatingDRho(x, y, dthetaInRadians);
		int drho_2 = drho_1 * 10;
		IncrementingHoughSpaceMatrixField(drho_2, rho, houghSpaceMatrix, theta, dtheta);

	}
}
void GoingThrewAllPossibleValuesOfDTheta(int x, int y, int rho, int theta, int* houghSpaceMatrix)
{
	for (int dtheta = 0; dtheta < 360; dtheta++)
	{
		float dthetaInRadians = (float)dtheta * pi / 180;
		int  drho = CalculatingDRho(x, y, dthetaInRadians);
		IncrementingHoughSpaceMatrixField(drho, rho, houghSpaceMatrix, theta, dtheta);
	}
}
void CalculatingRhoThetaMatrix(unsigned __int8* picture, int width, int height, int rho, int theta, int* houghSpaceMatrix)
{
	for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
			if (picture[y * width + x] == 255)
			{
				GoingThrewAllPossibleValuesOfDTheta(x, y, rho, theta, houghSpaceMatrix);
			}
}

int CalculatingMatrixMaximum(int rho, int theta, int* houghSpaceMatrix)
{
	int matrixMaximum = 0;
	for (int height = 0; height < rho; height++)
		for (int width = 0; width < theta; width++)
			if (houghSpaceMatrix[theta * height + width] > matrixMaximum)
			{
				matrixMaximum = houghSpaceMatrix[height * theta + width];
			}
	return matrixMaximum;
}
std::vector<int> FindingLocalMaximums(int rho, int theta, int* houghSpaceMatrix, int matrixMaximum, float cuttingFactor)
{
	std::vector<int> rhoThetaBeforeClipping;
	for (int height = 0; height < rho; height++)
		for (int width = 0; width < theta; width++)
		{
			if ((houghSpaceMatrix[theta * height + width] > (matrixMaximum * cuttingFactor)))
			{
				rhoThetaBeforeClipping.push_back(height);
				rhoThetaBeforeClipping.push_back(width);
			}
		}
	return rhoThetaBeforeClipping;
}
void AddALineToOutputArray(int* outputArray, int* arrayIncrement, int* rhoThetaBeforeClipping, int index)
{

	outputArray[*arrayIncrement] = rhoThetaBeforeClipping[index];
	outputArray[*(arrayIncrement)+1] = rhoThetaBeforeClipping[index + 1];
	*arrayIncrement += 2;
}
int TakeTheMiddleLineIndex(int i, int numberOfSimilarLines)
{
	int s = i - numberOfSimilarLines;
	if (s % 2 == 1 || (s == -1))
		s = s + 1;
	return s;
}
std::vector<int> Filtering(std::vector<int> rhoThetaBeforeClipping, int rhoDifference, int thetaDifference)
{
	int numberOfSimilarLines = 1;
	std::vector<int> outputArray;
	int i;
	for (i = 0; i < rhoThetaBeforeClipping.size() - 3; i += 2)
	{
		if ((rhoThetaBeforeClipping[i + 2] - rhoThetaBeforeClipping[i] < rhoDifference) && (abs(rhoThetaBeforeClipping[i + 3] - rhoThetaBeforeClipping[i + 1]) < thetaDifference))
			numberOfSimilarLines++;
		else
		{
			int index = TakeTheMiddleLineIndex(i, numberOfSimilarLines);
			outputArray.push_back(rhoThetaBeforeClipping[index]);
			outputArray.push_back(rhoThetaBeforeClipping[index + 1]);
			numberOfSimilarLines = 1;
		}
	}
	int index = TakeTheMiddleLineIndex(i, numberOfSimilarLines);
	outputArray.push_back(rhoThetaBeforeClipping[index]);
	outputArray.push_back(rhoThetaBeforeClipping[index + 1]);
	numberOfSimilarLines = 1;
	rhoThetaBeforeClipping.clear();
	return outputArray;
}
std::vector<int> doPicture(unsigned __int8* picture, int width, int height)
{

	int rho = sqrt(pow(width, 2) + pow(height, 2));
	int theta = 360;

	int* houghSpaceMatrix = new int[rho * theta]();

	CalculatingRhoThetaMatrix(picture, width, height, rho, theta, houghSpaceMatrix);

	int matrixMaximum = CalculatingMatrixMaximum(rho, theta, houghSpaceMatrix);

	//int SIZE = 1000;
	std::vector<int> rhoThetaBeforeClipping = FindingLocalMaximums(rho, theta, houghSpaceMatrix, matrixMaximum, 0.3);

	delete[] houghSpaceMatrix;

	return rhoThetaBeforeClipping;
}