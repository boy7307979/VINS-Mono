#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

void SADCalcCost(int meas_cnt, int num_disparity, float bf,
                 float* H, float* h, float* img_ref, float* img_cur,
                 size_t height, size_t width, size_t step, float* cost);

void SGM4PathCalcCost(float p1, float p2, float tau_so, float sgm_q1, float sgm_q2,
                      int num_disparity, size_t height, size_t width,
                      float* sad_cost, float* sgm_cost);

void Postprocessing(float* cost, size_t height, size_t width, int num_disparity,
                    float bf, float fx, float fy, float cx, float cy,
                    float* point_cloud, size_t point_cloud_step,
                    float* depth_map, size_t depth_map_step);

void WinnerTakesAllDisparity(float* cost, size_t height, size_t width, int num_disparity,
                             unsigned char* dis_mat, size_t dis_step);
