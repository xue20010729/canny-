//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <queue>
//#include <ctime>
//#include "opencv2/opencv.hpp"
//#define PI 3.1415926
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//    int row = 2 , col = 5;
//    float * vec =new float [row* col];
//    float ** matrix = new float *[row];
//    for(int i=0;i<row; i++){
//        matrix[i] = vec + i* col;
//    }
//    for(int i = 0;i<row;i++){
//        for (int j = 0; j < col; j++) {
//            matrix[i][j] = i * j;
//        }
//    }
//    Mat test(row,col,CV_32FC1,vec);
////    for(int i=0;i<row*col;i++){
////        cout<<vec[i]<<" ";
////    }
//    delete[] vec;
//    delete[] matrix;
//    return 0;
//}