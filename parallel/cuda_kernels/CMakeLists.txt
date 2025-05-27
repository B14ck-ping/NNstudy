if(NOT USE_CUDA)
    return()
endif()

cmake_minimum_required(VERSION 3.5) 

project(parallel_cuda LANGUAGES CUDA)

# add_library(foo STATIC matrix.cpp)# perceptron.cpp)#  matrixCUDA.cu)#processor.cpp
add_library(parallel_cuda STATIC kernels.cu) 
#processor.cpp

# set_target_properties(foo PROPERTIES CUDA_ARCHITECTURES "50;72")
set_target_properties(parallel_cuda PROPERTIES CUDA_ARCHITECTURES "50;72")
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(parallel_cuda PUBLIC cxx_std_17)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
# set_target_properties(foo
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parallel_cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        parallel_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


