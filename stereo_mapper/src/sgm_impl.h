#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

void SADCalcCost(int meas_cnt, int num_disparity, float bf,
                 float* H, float* h, float* img_ref, float* img_cur,
                 size_t height, size_t width, size_t step, float* cost);
