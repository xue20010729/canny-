//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <queue>
//#include <ctime>
//#include "opencv2/opencv.hpp"
//#include <emmintrin.h>
//#include <pmmintrin.h> //SSE3
//#include <immintrin.h>
//#define PI 3.1415926
//
//int main()
//{
//    int row =5 ,col =5;
//    float * dx_vec = new float [row * col];
//    float ** dx_matrix = new float *[row];
//    for(int i=0;i<row; i++){
//        dx_matrix[i] = dx_vec + i* col;
//    }
//    for(int i=0;i<row*col;i++){
//        dx_vec[i]=i;
//    }
//
//    for(int i=0;i<row*col;i++){
//        std::cout<<dx_vec[i]<<" ";
//    }
//    std::cout<<std::endl;
//    __m128 dx_float_vec =_mm_loadu_ps(dx_matrix[2]+0);
//    dx_float_vec = _mm_mul_ps(dx_float_vec,dx_float_vec);
//    _mm_storeu_ps(dx_matrix[2]+0,dx_float_vec);
//
//    for(int i=0;i<row*col;i++){
//        std::cout<<dx_vec[i]<<" ";
//    }
//    delete[] dx_vec;
//    delete[] dx_matrix;
//    return 0;
//}