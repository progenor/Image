cmake_minimum_required(VERSION 3.10)
project(Lab10)

find_package(OpenCV REQUIRED)

add_executable(Lab10 main.cpp)
target_link_libraries(Lab10 ${OpenCV_LIBS})