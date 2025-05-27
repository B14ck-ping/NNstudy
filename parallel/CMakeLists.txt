cmake_minimum_required(VERSION 3.18)
project(parallels LANGUAGES CXX)


# Общий заголовочный путь
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Подмодули по платформам
if(USE_CUDA)
    add_subdirectory(cuda_kernels)
endif()

if(USE_OPENCL)
    add_subdirectory(opencl)
endif()
