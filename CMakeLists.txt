cmake_minimum_required(VERSION 3.5) 

project(neuralNet LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# target_compile_definitions(neuralNet PUBLIC DUSE_CUDA)
# Опции включения бэкендов
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_OPENCL "Enable OpenCL support" OFF)

# Добавляем сабмодули
# add_subdirectory(common)
if(USE_CUDA)
    message(STATUS "CUDA support enabled")
    add_subdirectory(parallel/cuda_kernels)
endif()
add_subdirectory(math)
# add_subdirectory(neuralnet)
# add_subdirectory(tests)



set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO  "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}  -O2 -g -pg -DNDEBUG")

#processor.cpp
 
add_executable(main main.cpp)

target_link_libraries(main fooCUDA)	


