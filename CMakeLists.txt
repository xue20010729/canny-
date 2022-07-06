cmake_minimum_required(VERSION 3.21)
project(opencv)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "D:/opencv/mingw_build/install/x64/mingw/lib")
find_package(OpenCV  REQUIRED)
# 打印OpenCV版本
MESSAGE("OpenCV version: ${OpenCV_VERSION}")
# 添加include地址
#include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_executable(opencv main.cpp myCanny.cpp test.cpp try.cpp openmp.cpp pthread.cpp)
# 添加libs地址
target_link_libraries(opencv ${OpenCV_LIBS})


