# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:56:32 on Thu, Jun 02, 2022
#
# Description: cmake for matrix multiply

cmake_minimum_required (VERSION 3.8)

project (matrix_multiply LANGUAGES CXX)

set (CMAKE_VERBOSE_MAKEFILE ${MATRIX_VERBOSE_MAKEFILE})

set (CMAKE_CXX_FLAGS "-std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DMATRIX_BUILD_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

add_compile_options (
    -Wall
    -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
    -ggdb
    -g
)

include_directories (
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cpu
)

# cpu
add_executable (matrix_multiply_cpu ${PROJECT_SOURCE_DIR}/src/cpu/cpu.cpp)

install (TARGETS matrix_multiply_cpu RUNTIME DESTINATION cpu)