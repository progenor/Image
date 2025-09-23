cmake_minimum_required(VERSION 3.10)
project(Lab2)

find_package(OpenCV REQUIRED)

add_executable(Lab2 main.cpp)
target_link_libraries(Lab2 ${OpenCV_LIBS})