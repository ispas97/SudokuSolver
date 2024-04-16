#include "pch.h"
#include<iostream>

extern "C"
{

	__declspec(dllexport) int add(int a, int b)
	{
		return a + b;
	}

	 __declspec(dllexport) void doPicture(unsigned __int8* image,int  height, int width)
	 {
		 int index;

		 for (int row = 0; row < height; row++)
		 {
			 for (int column = 0; column < width; column++)
			 {
				 index = (row * width + column)*3;
				 image[index] = 0;
				 image[index + 2] = 0;
			 }
		 }
	 }

};
