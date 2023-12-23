cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(pytorch_cpp)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "/Users/kapilsharma/oss/pytorch/torch")

message(${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pytorch_cpp main.cpp)
target_link_libraries(pytorch_cpp "${TORCH_LIBRARIES}")
