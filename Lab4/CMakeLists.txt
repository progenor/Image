cmake_minimum_required(VERSION 3.10)
project(Lab4)

find_package(OpenCV REQUIRED)

add_executable(Lab4 main.cpp)
target_link_libraries(Lab4 ${OpenCV_LIBS})