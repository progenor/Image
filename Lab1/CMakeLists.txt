cmake_minimum_required(VERSION 3.10)
project(Lab1)

find_package(OpenCV REQUIRED)

add_executable(Lab1 main.cpp)
target_link_libraries(Lab1 ${OpenCV_LIBS})