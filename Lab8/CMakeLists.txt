cmake_minimum_required(VERSION 3.10)
project(Lab8)

find_package(OpenCV REQUIRED)

add_executable(Lab8 main.cpp)
target_link_libraries(Lab8 ${OpenCV_LIBS})