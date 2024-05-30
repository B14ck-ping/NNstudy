cmake_minimum_required(VERSION 3.0) 

# set(CMAKE_C_STANDARD 11)

project(neuralNet)

add_library(foo STATIC matrix.cpp)
			
add_executable(main main.cpp)	

target_link_libraries(main foo)	