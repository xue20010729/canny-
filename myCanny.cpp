//// 针对gradient函数做了SIMD优化
//
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
//using namespace std;
//using namespace cv;
//
//class my_Canny
//{
//public:
//    int kernel, thresh_low, thresh_high, max_edge = 512;
//    float sigma;
//    bool interactive;
//    int row,col;
//    my_Canny(int kernel = 5, int thresh_low = 50, int thresh_high = 200, bool interactive = false, float sigma = 1.4) {
//        this->kernel = kernel, this->thresh_low = thresh_low, this->thresh_high = thresh_high;
//        this->interactive = interactive, this->sigma = sigma;
//    }
//    Mat read_image_and_convert(const string image_name); //读取图像并转为最大边512的图像
//    Mat smooth(const Mat &image); //搭配convolution函数进行高斯模糊处理
//    Mat convolution(const Mat &src, const Mat &kernel); // 卷积操作
//
//    void gradient_sse(const Mat &image, Mat &dx, Mat &dy, Mat &magnitude);
//    Mat hist_equalize(const Mat &image);
//
//    Mat nms(const Mat &dx, const Mat &dy, const Mat &magnitude);
//
//    Mat double_threshold(const Mat &nms);
//
//    void check_iterative(const Mat &nms, Mat &boundary, Mat &checked, queue<vector<int>> &lists);
//
//    void copy_to_matrix(float **matrix, const Mat &img);
//};
//
//Mat my_Canny::read_image_and_convert(const string name) {
//    Mat image =imread(name, IMREAD_COLOR);
//    if (max(image.rows, image.cols) > this->max_edge) {
//        float ratio = this->max_edge * 1. / max(image.rows, image.cols);
//        resize(image, image, Size((int)image.cols * ratio, (int)image.rows * ratio));
//    }
//    cvtColor(image, image, COLOR_BGR2GRAY); // 转换为灰度图
//    image.convertTo(image, CV_32FC1);  // 换为32位float
//    this -> row = image.rows;
//    this -> col = image.cols;
//    return image;
//}
//
//Mat my_Canny::smooth(const Mat& image) {
//    Mat gaussian_kernel(this->kernel, this->kernel, CV_32FC1);
//    float coef = 1. / 2 / PI / this->sigma / this->sigma;
//    for (int row = 0; row < gaussian_kernel.rows; row++) {
//        for (int col = 0; col < gaussian_kernel.cols; col++) {
//            gaussian_kernel.at<float>(row, col) = exp(-(pow(row - kernel / 2, 2) + pow(col - kernel / 2, 2)) / 2 / pow(sigma, 2)) * coef;
//        }
//    }
//    return this->convolution(image, gaussian_kernel);
//}
//
//Mat my_Canny::convolution(const Mat& src, const Mat& kernel) {
//    Mat blurred(src.rows, src.cols, CV_32FC1, Scalar(0));
//    int kernel_size = kernel.rows;
//    for (int row = 0; row < src.rows - kernel_size; row++) {
//        for (int col = 0; col < src.cols - kernel_size; col++) {
//            float total = 0;
//            for (int i = 0; i < kernel_size; i++) {
//                for (int j = 0; j < kernel_size; j++) {
//                    total += kernel.at<float>(i, j) * src.at<float>(row + i, col + j);
//                }
//            }
//            blurred.at<float>(row + kernel_size / 2, col + kernel_size / 2) = total;
//        }
//    }
//    return blurred;
//}
//
//Mat my_Canny::hist_equalize(const Mat& image) {
//    Mat new_image(image.rows, image.cols, CV_32FC1, Scalar(0));
//    Mat uint_image;
//    image.convertTo(uint_image, CV_8UC1);
//    vector<int> count(256, 0);
//    vector<float> s(256, 0);
//    for (int i = 0; i < image.rows; i++) {
//        for (int j = 0; j < image.cols; j++) {
//            count[uint_image.at<uint8_t>(i, j)]++;
//        }
//    }
//    int sigma = 0;
//    float zero_count = count[0];
//    count[0] = 0;
//    for (int i = 0; i < 256; i++) {
//        sigma += count[i];
//        s[i] = 255. * sigma * 1. / max(uint_image.rows * uint_image.cols - zero_count, 1.f);
//    }
//    for (int i = 0; i < image.rows; i++) {
//        for (int j = 0; j < image.cols; j++) {
//            new_image.at<float>(i, j) = s[uint_image.at<uint8_t>(i, j)];
//        }
//    }
//    return new_image;
//}
//
//void my_Canny::gradient_sse(const Mat& image, Mat& dx, Mat& dy, Mat& magnitude) {
//    float kernel_x_array[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
//    float kernel_y_array[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
//    Mat kernel_x(3, 3, CV_32FC1, &kernel_x_array);
//    Mat kernel_y(3, 3, CV_32FC1, &kernel_y_array);
//    dx = this->convolution(image, kernel_x);
//    dy = this->convolution(image, kernel_y);
//
//    float * dx_vec = new float [this->row * this->col];
//    float ** dx_matrix = new float *[this->row];
//    for(int i=0;i<row; i++){
//        dx_matrix[i] = dx_vec + i* col;
//    }
//    float * dy_vec = new float [this->row * this->col];
//    float ** dy_matrix = new float *[this->row];
//    for(int i=0;i<row; i++){
//        dy_matrix[i] = dx_vec + i* col;
//    }
//    copy_to_matrix(dx_matrix,dx);
//    copy_to_matrix(dy_matrix,dy);
//
//    float * magnitude_vec = new float [this->row * this->col];
//    float ** magnitude_matrix = new float *[this->row];
//    for(int i=0;i<row; i++){
//        magnitude_matrix[i] = magnitude_vec + i* col;
//    }
//
//    float max_magnitude = 0;
////    for (int i = 0; i < image.rows; i++) {
////        for (int j = 0; j < image.cols; j++) {
////            magnitude_matrix[i][j] = sqrt(pow(dx_matrix[i][j], 2) + pow(dy_matrix[i][j], 2));
////        }
////    }
//    for(int i =0 ;i<image.rows;i++){
//        int j;
//        for(j=0; j+4 < image.cols; j+=4){
//            __m128 dx_float_vec =_mm_loadu_ps(dx_matrix[i]+j);
//            __m128 dy_float_vec =_mm_loadu_ps(dy_matrix[i]+j);
//            dx_float_vec = _mm_mul_ps(dx_float_vec,dx_float_vec);
//            dy_float_vec = _mm_mul_ps(dy_float_vec,dy_float_vec);
//            __m128 res = _mm_sqrt_ps(_mm_add_ps(dx_float_vec,dy_float_vec));
//            _mm_storeu_ps(magnitude_matrix[i]+j,res);
//        }
//        for(;j<image.cols;j++){
//            magnitude_matrix[i][j] = sqrt(pow(dx_matrix[i][j], 2) + pow(dy_matrix[i][j], 2));
//        }
//    }
//    magnitude = Mat(this->row, this->col,CV_32FC1,magnitude_vec);
//    magnitude = this->hist_equalize(magnitude);
//    delete[] dx_vec;
//    delete[] dx_matrix;
//    delete[] dy_vec;
//    delete[] dy_matrix;
//    delete[] magnitude_vec;
//    delete[] magnitude_matrix;
//}
//
//void my_Canny::copy_to_matrix(float** matrix,const Mat &img) {
//    for(int i=0;i<this->row;i++){
//        for(int j=0;j<this->col;j++)
//            matrix[i][j] = img.at<float>(i,j);
//    }
//}
////非极大值抑制
//Mat my_Canny::nms(const Mat& dx, const Mat& dy, const Mat& magnitude) {
//    float grad_x, grad_y, grad, grad1, grad2, grad3, grad4, grad_temp1, grad_temp2, weight;
//    int direction;
//    Mat nms = magnitude.clone();
//    int boundary = this->kernel;
//    for (int i = 0; i < nms.rows; i++) {
//        for (int j = 0; j < nms.cols; j++) {
//            if (i < boundary || j < boundary || i >= nms.rows - boundary - 1 || j >= nms.cols - boundary - 1) {
//                nms.at<float>(i, j) = 0;
//            }
//        }
//    }
//    for (int i = 1; i < nms.rows - 1; i++) {
//        for (int j = 1; j < nms.cols - 1; j++) {
//            if (magnitude.at<float>(i, j) > this->thresh_low) {
//                grad_x = dx.at<float>(i, j);
//                grad_y = dy.at<float>(i, j);
//                grad = magnitude.at<float>(i, j);
//                direction = ((int)grad_x * grad_y > 0) * 2 - 1;
//                if (abs(grad_y) > abs(grad_x)) {
//                    weight = abs(grad_x) / max(abs(grad_y), 1e-6f);
//                    grad2 = magnitude.at<float>(i - 1, j);
//                    grad4 = magnitude.at<float>(i + 1, j);
//
//                    grad1 = magnitude.at<float>(i - 1, j - direction);
//                    grad3 = magnitude.at<float>(i + 1, j + direction);
//                }
//                else {
//                    weight = abs(grad_y) / max(abs(grad_x), 1e-6f);
//                    grad2 = magnitude.at<float>(i, j - 1);
//                    grad4 = magnitude.at<float>(i, j + 1);
//
//                    grad1 = magnitude.at<float>(i - direction, j - 1);
//                    grad3 = magnitude.at<float>(i + direction, j + 1);
//                }
//                grad_temp1 = (1 - weight) * grad1 + weight * grad2;
//                grad_temp2 = (1 - weight) * grad3 + weight * grad4;
//                if (grad <= grad_temp1 || grad < grad_temp2) {
//                    nms.at<float>(i, j) = 0;
//                }
//            }
//        }
//    }
//    return nms;
//}
//
//
////查中间阈值是否连通
//void my_Canny::check_iterative(const Mat& nms, Mat& boundary, Mat& checked, queue<vector<int> >& lists) {
//    int i, j;
//    vector<int> coor;
//    while (!lists.empty()) {
//        i = lists.front()[0];
//        j = lists.front()[1];
//        lists.pop();
//        if (i < 0 || j < 0 || i >= boundary.rows || j >= boundary.cols)
//            continue;
//        if (checked.at<uint8_t>(i, j))
//            continue;
//        checked.at<uint8_t>(i, j) = 1;
//        if (nms.at<float>(i, j) <= this->thresh_low)
//            continue;
//        boundary.at<uint8_t>(i, j) = 255;
//        for (int row : {-1, 0, 1}) {
//            for (int col : {-1, 0, 1}) {
//                if (row || col)
//                    lists.push(vector<int>{i + row, j + col});
//            }
//        }
//    }
//}
////双阈值
//Mat my_Canny::double_threshold(const Mat& nms) {
//    Mat boundary(nms.rows, nms.cols, CV_8UC1, Scalar(0));
//    Mat checked(nms.rows, nms.cols, CV_8UC1, Scalar(0));
//    Mat new_nms = nms.clone();
//    for (int i = 0; i < new_nms.rows; i++) {
//        for (int j = 0; j < new_nms.cols; j++) {
//            if (new_nms.at<float>(i, j) < this->thresh_low) {
//                new_nms.at<float>(i, j) = 0;
//            }
//        }
//    }
//    new_nms = this->hist_equalize(new_nms);
//    queue<vector<int> > lists;
//    for (int i = 0; i < new_nms.rows; i++) {
//        for (int j = 0; j < new_nms.cols; j++) {
//            if (new_nms.at<float>(i, j) > this->thresh_high) {
//                lists.push(vector<int>{i, j});
//            }
//        }
//    }
//    this->check_iterative(new_nms, boundary, checked, lists);
//    return boundary;
//}
//
//int main()
//{
//    my_Canny canny(5, 100, 200, false, 1.4);
//    clock_t start, end;
//    start = clock();
//
//    Mat src =canny.read_image_and_convert("../2.jpg");
////    src = canny.smooth(src);
//    Mat dx, dy, magnitude;
//    canny.gradient_sse(src, dx, dy, magnitude);
//    Mat nms = canny.nms(dx, dy, magnitude);
//    Mat thresh = canny.double_threshold(nms);
//    Mat disp_img;
//    thresh.convertTo(disp_img, CV_8UC1);
//
//    end = clock();
//    cout << "my version time used: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
//
//    imshow("my version", disp_img);
//    waitKey(0);
//    return 0;
//}