#pragma once
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;


	int* rhoThetaMatrixInitilization(int rho, int theta);
	float CalculatingDRho(int x, int y, float dthetaInRadians);
	void IncrementingHoughSpaceMatrixField(int drho, int rho, int* houghSpaceMatrix, int theta, int dtheta);
	void GoingThrewAllPossibleValuesOfDTheta2(int x, int y, int rho, int theta, int* houghSpaceMatrix);
	void GoingThrewAllPossibleValuesOfDTheta(int x, int y, int rho, int theta, int* houghSpaceMatrix);
	void CalculatingRhoThetaMatrix(unsigned __int8* picture, int width, int height, int rho, int theta, int* houghSpaceMatrix);
	int CalculatingMatrixMaximum(int rho, int theta, int* houghSpaceMatrix);
	std::vector<int> FindingLocalMaximums(int rho, int theta, int* houghSpaceMatrix, int matrixMaximum, float cuttingFactor = 0.5);
	void AddALineToOutputArray(int* outputArray, int* arrayIncrement, int* rhoThetaBeforeClipping, int index);
	int TakeTheMiddleLineIndex(int i, int numberOfSimilarLines);
	std::vector<int> Filtering(std::vector<int> rhoThetaBeforeClipping, int rhoDifference = 15, int thetaDifference = 5);
	std::vector<int> doPicture(unsigned __int8* picture, int width, int height);
