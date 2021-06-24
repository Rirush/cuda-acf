cmake_minimum_required(VERSION 3.19)
project(cuda_signal_analysis CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_signal_analysis main.cu)

set_target_properties(
        cuda_signal_analysis
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)