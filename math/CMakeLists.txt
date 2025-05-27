cmake_minimum_required(VERSION 3.5) 

project(neuralNet LANGUAGES CXX)
# set(CUDAHOSTCXX g++)

if(USE_CUDA)
    target_compile_definitions(math PRIVATE DUSE_CUDA)
    target_link_libraries(math PRIVATE cuda_kernels)
endif()


