cmake_minimum_required(VERSION 3.0) 

set(CMAKE_CPP_STANDARD 17)

project(neuralNet)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

SET(GCC_COVERAGE_COMPILE_FLAGS "-march=native -pg -msse4.1 -O3 -Wall")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_library(foo STATIC matrix.cpp perceptron.cpp processor.cpp)
			
add_executable(main main.cpp)	

target_link_libraries(main foo)	