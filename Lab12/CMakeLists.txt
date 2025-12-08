cmake_minimum_required(VERSION 3.10)
project(Lab12)

find_package(OpenCV REQUIRED)

add_executable(Lab12 main.cpp)
target_link_libraries(Lab12 ${OpenCV_LIBS})