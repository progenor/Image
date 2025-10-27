cmake_minimum_required(VERSION 3.10)
project(Lab7)

find_package(OpenCV REQUIRED)

add_executable(Lab7 main.cpp)
target_link_libraries(Lab7 ${OpenCV_LIBS})