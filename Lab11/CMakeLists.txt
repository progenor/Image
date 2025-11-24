cmake_minimum_required(VERSION 3.10)
project(Lab11)

find_package(OpenCV REQUIRED)

add_executable(Lab11 main.cpp)
target_link_libraries(Lab11 ${OpenCV_LIBS})