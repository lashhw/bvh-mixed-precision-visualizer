cmake_minimum_required(VERSION 3.23)
project(bvh_mixed_precision)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(bvh)

add_executable(bvh_mixed_precision main.cpp)
target_link_libraries(bvh_mixed_precision PUBLIC bvh mpfr)