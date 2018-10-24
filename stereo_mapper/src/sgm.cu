#include "sgm_impl.h"
#include <cassert>

static texture<float, cudaTextureType2D, cudaReadModeElementType> tex2d_ref;
static texture<float, cudaTextureType2D, cudaReadModeElementType> tex2d_cur;

#define INDEX(v, u, d, width, nd) ((v*width+u)*nd + d)

__global__ void SADCalcCostKernel(int meas_cnt, int num_disparity, float bf,
                                  float* H, float* h, size_t height, size_t width,
                                  float* cost) {
    // blockDim.x = num_disparity
    // gridDim.x = width
    // gridDim.y = height
    // blockIdx.x = [0, width - 1]
    // blockIdx.y = [0, height - 1]
    // threadIdx.x = [0, num_disparity - 1]
    const int p_u = blockIdx.x;
    const int p_v = blockIdx.y;
    const int disparity = threadIdx.x;

    // H = [H[0] H[1] H[2],
    //      H[3] H[4] H[5],
    //      H[6] H[7] H[8]]

    // h = [h[0] h[1] h[2]]'
    float Hp[3];
    Hp[0] = H[0] * p_u + H[1] * p_v + H[2];
    Hp[1] = H[3] * p_u + H[4] * p_v + H[5];
    Hp[2] = H[6] * p_u + H[7] * p_v + H[8];

    float Hpu[3]; // (0, -1)
    Hpu[0] = Hp[0] - H[1];
    Hpu[1] = Hp[1] - H[4];
    Hpu[2] = Hp[2] - H[7];

    float Hpd[3]; // (0, 1)
    Hpd[0] = Hp[0] + H[1];
    Hpd[1] = Hp[1] + H[4];
    Hpd[2] = Hp[2] + H[7];

    float Hpl[3]; // (-1, 0)
    Hpl[0] = Hp[0] - H[0];
    Hpl[1] = Hp[1] - H[3];
    Hpl[2] = Hp[2] - H[6];

    float Hpr[3]; // (1, 0)
    Hpr[0] = Hp[0] + H[0];
    Hpr[1] = Hp[1] + H[3];
    Hpr[2] = Hp[2] + H[6];

    float Hpul[3]; // (0, -1) + (-1, 0)
    Hpul[0] = Hpu[0] - H[0];
    Hpul[1] = Hpu[1] - H[3];
    Hpul[2] = Hpu[2] - H[6];

    float Hpur[3];
    Hpur[0] = Hpu[0] + H[0];
    Hpur[1] = Hpu[1] + H[3];
    Hpur[2] = Hpu[2] + H[6];

    float Hpdl[3];
    Hpdl[0] = Hpd[0] - H[0];
    Hpdl[1] = Hpd[1] - H[3];
    Hpdl[2] = Hpd[2] - H[6];

    float Hpdr[3];
    Hpdr[0] = Hpd[0] + H[0];
    Hpdr[1] = Hpd[1] + H[3];
    Hpdr[2] = Hpd[2] + H[6];

    float *cost_ptr = cost + INDEX(p_u, p_v, disparity, width, num_disparity);
    float last_cost = (meas_cnt == 1)? 0 : *cost_ptr;
    if(meas_cnt != 1 && last_cost < 0)
        return;

    float cost_value = 0.0f;
    float inv_depth = disparity / bf;
#define PROJECT(H, uu, vv) {\
            float z = H[2] + h[2] * inv_depth; \
            float ppu = (H[0] + h[0] * inv_depth) / z; \
            float ppv = (H[1] + h[1] * inv_depth) / z; \
            if(ppu < 0 || ppv < 0 || ppu >= width || ppv >= height) { \
                *cost_ptr = -1.0f; \
                return; \
            } \
            cost_value += fabs(tex2D(tex2d_ref, p_u + 0.5 + uu, p_v + 0.5 + vv) - \
                               tex2D(tex2d_cur, ppu + 0.5, ppv + 0.5)); \
        }
    PROJECT(Hp, 0, 0);
    PROJECT(Hpu, 0, -1);
    PROJECT(Hpd, 0, 1);
    PROJECT(Hpl, -1, 0);
    PROJECT(Hpr, 1, 0);
    PROJECT(Hpul, -1, -1);
    PROJECT(Hpur, 1, -1);
    PROJECT(Hpdl, -1, 1);
    PROJECT(Hpdr, 1, 1);
#undef PROJECT

    if(meas_cnt == 1)
        *cost_ptr = cost_value / 9.0f;
    else
        *cost_ptr = (last_cost * (meas_cnt - 1) + cost_value / 9.0f) / meas_cnt;
}

void SADCalcCost(int meas_cnt, int num_disparity, float bf,
                 float* h_H, float* h_h, float* img_ref, float* img_cur,
                 size_t height, size_t width, size_t step, float* cost) {
    cudaUnbindTexture(tex2d_cur);
    cudaUnbindTexture(tex2d_ref);

    dim3 grid = dim3(width, height);
    dim3 block = dim3(num_disparity);

    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<float>();
    tex2d_ref.addressMode[0] = cudaAddressModeBorder;
    tex2d_ref.addressMode[1] = cudaAddressModeBorder;
    tex2d_ref.filterMode = cudaFilterModeLinear;
    tex2d_ref.normalized = false;
    tex2d_cur.addressMode[0] = cudaAddressModeBorder;
    tex2d_cur.addressMode[1] = cudaAddressModeBorder;
    tex2d_cur.filterMode = cudaFilterModeLinear;
    tex2d_cur.normalized = false;

    size_t offset = 0;
    cudaBindTexture2D(&offset, tex2d_ref, img_ref, ca_desc0, width, height, step);
    assert(offset == 0);
    cudaBindTexture2D(&offset, tex2d_cur, img_cur, ca_desc1, width, height, step);
    assert(offset == 0);

    float *d_H, *d_h;
    cudaMalloc(&d_H, 9 * sizeof(float));
    cudaMalloc(&d_h, 3 * sizeof(float));

    cudaMemcpy(d_H, h_H, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, 3 * sizeof(float), cudaMemcpyHostToDevice);

    SADCalcCostKernel<<<grid, block>>>(meas_cnt, num_disparity, bf,
                                       d_H, d_h, height, width, cost);

    cudaFree(d_H);
    cudaFree(d_h);
}
#undef INDEX
