cmake_minimum_required(VERSION 3.10)
project(Lab5)

find_package(OpenCV REQUIRED)

add_executable(Lab5 main.cpp)
target_link_libraries(Lab5 ${OpenCV_LIBS})