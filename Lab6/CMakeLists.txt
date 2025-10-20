cmake_minimum_required(VERSION 3.10)
project(Lab6)

find_package(OpenCV REQUIRED)

add_executable(Lab6 main.cpp)
target_link_libraries(Lab6 ${OpenCV_LIBS})