cmake_minimum_required(VERSION 3.10)
project(Lab9)

find_package(OpenCV REQUIRED)

add_executable(Lab9 main.cpp)
target_link_libraries(Lab9 ${OpenCV_LIBS})