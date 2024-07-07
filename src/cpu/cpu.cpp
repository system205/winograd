// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:56:32 on Thu, Jun 02, 2022
//
// Description: test matrix multiply on cpu

#include "common.h"
#include "cpu_coppersmith_winograd.h"

#define CPU_MATRIX_DIMENSION 64

int main() {
    MATRIX_TRACE_PROFILE("matrix_multiply_cpu");

    size_t A_row = CPU_MATRIX_DIMENSION, A_column = CPU_MATRIX_DIMENSION, B_row = CPU_MATRIX_DIMENSION,
           B_column = CPU_MATRIX_DIMENSION;
    std::vector<std::vector<float>> A(A_row, std::vector<float>(A_column));
    std::vector<std::vector<float>> B(B_row, std::vector<float>(B_column));

    float min = 1.0, max = 100.0;
    get_random_matrix<float>(A, min, max);
    // print_matrix<float>(A, "A");

    get_random_matrix<float>(B, min, max);
    // print_matrix<float>(B, "B");

    std::vector<std::vector<float>> C(A_row, std::vector<float>(B_column));

    MLOG("=================== coppersmith-winograd ===================");
    matrix_multiply_cpu_coppersmith_winograd<float>(A, B, C);
    // print_matrix<float>(C, "coppersmith-winograd C");

    return 0;
}
