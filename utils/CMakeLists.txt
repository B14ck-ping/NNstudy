cmake_minimum_required(VERSION 3.5) 

project(neuralNet LANGUAGES CXX CUDA )
# set(CUDAHOSTCXX g++)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO  "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}  -O2 -g -pg -DNDEBUG")


# add_library(foo STATIC matrix.cpp)# perceptron.cpp)#  matrixCUDA.cu)#processor.cpp
add_library(fooCUDA STATIC  matrixCUDA.cu matrix.cpp perceptron.cu kernels.cu) 
#processor.cpp

# set_target_properties(foo PROPERTIES CUDA_ARCHITECTURES "50;72")
set_target_properties(fooCUDA PROPERTIES CUDA_ARCHITECTURES "50;72")
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(fooCUDA PUBLIC cxx_std_17)
 
add_executable(main main.cpp)


# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
# set_target_properties(foo
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fooCUDA
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fooCUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "50;72")
# target_link_libraries(foo fooCUDA)	
target_link_libraries(main fooCUDA)	


