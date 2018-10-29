#include "sgm_impl.h"
#include <cassert>
#include <cfloat>

static texture<float, cudaTextureType2D, cudaReadModeElementType> tex2d_ref;
static texture<float, cudaTextureType2D, cudaReadModeElementType> tex2d_cur;

#define INDEX(v, u, d, width, nd) ((v*width+u)*nd+d)

__global__ void SADCalcCostKernel(int meas_cnt, int num_disparity, float bf,
                                  float H0, float H1, float H2,
                                  float H3, float H4, float H5,
                                  float H6, float H7, float H8,
                                  float h0, float h1, float h2,
                                  size_t height, size_t width,
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
    float Hp_x, Hp_y, Hp_z;
    Hp_x = H0 * p_u + H1 * p_v + H2;
    Hp_y = H3 * p_u + H4 * p_v + H5;
    Hp_z = H6 * p_u + H7 * p_v + H8;

    float Hpu_x, Hpu_y, Hpu_z; // (0, -1)
    Hpu_x = Hp_x - H1;
    Hpu_y = Hp_y - H4;
    Hpu_z = Hp_z - H7;

    float Hpd_x, Hpd_y, Hpd_z; // (0, 1)
    Hpd_x = Hp_x + H1;
    Hpd_y = Hp_y + H4;
    Hpd_z = Hp_z + H7;

    float Hpl_x, Hpl_y, Hpl_z; // (-1, 0)
    Hpl_x = Hp_x - H0;
    Hpl_y = Hp_y - H3;
    Hpl_z = Hp_z - H6;

    float Hpr_x, Hpr_y, Hpr_z; // (1, 0)
    Hpr_x = Hp_x + H0;
    Hpr_y = Hp_y + H3;
    Hpr_z = Hp_z + H6;

    float Hpul_x, Hpul_y, Hpul_z; // (0, -1) + (-1, 0)
    Hpul_x = Hpu_x - H0;
    Hpul_y = Hpu_y - H3;
    Hpul_z = Hpu_z - H6;

    float Hpur_x, Hpur_y, Hpur_z;
    Hpur_x = Hpu_x + H0;
    Hpur_y = Hpu_y + H3;
    Hpur_z = Hpu_z + H6;

    float Hpdl_x, Hpdl_y, Hpdl_z;
    Hpdl_x = Hpd_x - H0;
    Hpdl_y = Hpd_y - H3;
    Hpdl_z = Hpd_z - H6;

    float Hpdr_x, Hpdr_y, Hpdr_z;
    Hpdr_x = Hpd_x + H0;
    Hpdr_y = Hpd_y + H3;
    Hpdr_z = Hpd_z + H6;

    float *cost_ptr = cost + INDEX(p_v, p_u, disparity, width, num_disparity);
    if (meas_cnt == 1 && (p_u == 0 || p_u == width - 1 || p_v == 0 || p_v == height - 1))
    {
        *cost_ptr = -1.0f;
        return;
    }

    float last_cost = *cost_ptr;
    if(meas_cnt != 1 && last_cost < 0)
        return;

    float cost_value = 0.0f;
    float inv_depth = disparity / bf;
#define PROJECT(HH, uu, vv) {\
            float z = HH##_z + h2 * inv_depth; \
            float ppu = (HH##_x + h0 * inv_depth) / z; \
            float ppv = (HH##_y + h1 * inv_depth) / z; \
            if(z < 0 || ppu < 0 || ppv < 0 || ppu >= width || ppv >= height) { \
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

    SADCalcCostKernel<<<grid, block>>>(meas_cnt, num_disparity, bf, h_H[0],
            h_H[1],h_H[2],h_H[3],h_H[4],h_H[5],h_H[6],h_H[7],h_H[8],h_h[0],
            h_h[1],h_h[2], height, width, cost);

    cudaDeviceSynchronize();
}

__global__ void SGMCalcCostKernel(int num_disparity, size_t height, size_t width, int idx,
                                  int start, int dx, int dy, int end, float p1, float p2, float tau_so,
                                  float sgm_q1, float sgm_q2, float* sad_cost, float* sgm_cost) {
    int xy[2] = {(int)blockIdx.x, (int)blockIdx.x};
    xy[idx] = start;
    int p_u = xy[0], p_v = xy[1];
    int d = threadIdx.x;

    __shared__ float output_s[128];
    __shared__ float output_min[128];
    __shared__ float input_s[128];
    __shared__ float input_min[128];

    input_s[d] = input_min[d] = sad_cost[INDEX(p_v, p_u, d, width, num_disparity)];
    __syncthreads();
    // find input_s min
    for(int i = num_disparity/2 ; i > 0; i /= 2) {
        if(d < i && d + i < num_disparity && input_min[d + i] < input_min[d])
            input_min[d] = input_min[d + i];
        __syncthreads();
    }

    if(input_min[0] < 0.0f) {
        input_s[d] = 0.0f;
        sgm_cost[INDEX(p_v, p_u, d, width, num_disparity)] = input_s[d];
        output_s[d] = output_min[d] = input_s[d];
    }
    else {
        sgm_cost[INDEX(p_v, p_u, d, width, num_disparity)] += input_s[d];
        output_s[d] = output_min[d] = input_s[d];
    }
    xy[0] += dx;
    xy[1] += dy;

    for(int k = 1; k < end; ++k, xy[0] += dx, xy[1] += dy) {
        p_u = xy[0];
        p_v = xy[1];

        input_s[d] = input_min[d] = sad_cost[INDEX(p_v, p_u, d, width, num_disparity)];
        __syncthreads();

        for(int i = num_disparity/2; i > 0; i /= 2) {
            if(d < i && d + i < num_disparity) {
                if(output_min[d + i] < output_min[d])
                    output_min[d] = output_min[d + i];
                if(input_min[d + i] < input_min[d])
                    input_min[d] = input_min[d + i];
            }
            __syncthreads();
        }

        if(input_min[0] < 0.0f) {
            input_s[d] = 0.0f;
            __syncthreads();
        }

        float G = fabs(tex2D(tex2d_ref, p_u + 0.5, p_v + 0.5) -
                       tex2D(tex2d_ref, p_u - dx + 0.5, p_v - dy + 0.5));
        float P1 = p1, P2 = p2;
        if(G <= tau_so) {
            P1 *= sgm_q1;
            P2 *= sgm_q2;
        }

        float cost = min(output_s[d], output_min[0] + P2);
        if(d - 1 >=0)
            cost = min(cost, output_s[d - 1] + P1);
        if(d + 1 < num_disparity)
            cost = min(cost, output_s[d + 1] + P1);

        float val = input_s[d] + cost - output_min[0];
        if(input_min[0] < 0.0f)
            sgm_cost[INDEX(p_v, p_u, d, width, num_disparity)] = 0.0;
        else
            sgm_cost[INDEX(p_v, p_u, d, width, num_disparity)] += val;

        __syncthreads();
        output_min[d] = output_s[d] = val;
        __syncthreads();
    }
}

void SGM4PathCalcCost(float p1, float p2, float tau_so, float sgm_q1,
                      float sgm_q2, int num_disparity, size_t height,
                      size_t width, float* sad_cost, float* sgm_cost) {
    SGMCalcCostKernel<<<height, num_disparity>>>(num_disparity, height, width,
                                                  0, 0, 1, 0, width, p1, p2,
                                                 tau_so, sgm_q1, sgm_q2, sad_cost,
                                                 sgm_cost);
    SGMCalcCostKernel<<<height, num_disparity>>>(num_disparity, height, width, 0,
                                                  width - 1, -1, 0, width, p1, p2,
                                                 tau_so, sgm_q1, sgm_q2, sad_cost,
                                                 sgm_cost);
    SGMCalcCostKernel<<<width, num_disparity>>>(num_disparity, height, width,
                                                  1, 0, 0, 1, height, p1, p2,
                                                 tau_so, sgm_q1, sgm_q2, sad_cost,
                                                 sgm_cost);
    SGMCalcCostKernel<<<width, num_disparity>>>(num_disparity, height, width, 1,
                                                  height - 1, 0, -1, height, p1, p2,
                                                 tau_so, sgm_q1, sgm_q2, sad_cost,
                                                 sgm_cost);
    cudaDeviceSynchronize();
}

__global__ void PostprocessingKernel(float* cost, size_t height, size_t width, int num_disparity,
                                     float bf, float fx, float fy, float cx, float cy,
                                     float* point_cloud, size_t point_cloud_step,
                                     float* depth_map, size_t depth_map_step) {
    const int p_u = blockIdx.x;
    const int p_v = blockIdx.y;
    const int d = threadIdx.x;

    float* point_cloud_ptr = point_cloud + (p_v * (point_cloud_step/4) + p_u * 3);
    float* depth_map_ptr = depth_map + p_v * (depth_map_step)/4 + p_u;

    __shared__ float c[128], c_min[128];
    __shared__ int c_idx[128];
    c[d] = c_min[d] = cost[INDEX(p_v, p_u, d, width, num_disparity)];
    c_idx[d] = d;
    __syncthreads();
    for(int i = num_disparity/2; i > 0; i /= 2) {
        if(d < i && d + i < num_disparity && c_min[d + i] < c_min[d]) {
            c_min[d] = c_min[d + i];
            c_idx[d] = c_idx[d + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        float min_cost = c_min[0];
        int min_idx = c_idx[0];

        if(min_cost <= 0 || min_idx == 0 || min_idx == num_disparity - 1
                || c[min_idx - 1] + c[min_idx + 1] < 2 * min_cost) {
            point_cloud_ptr[0] = -1;
            point_cloud_ptr[1] = -1;
            point_cloud_ptr[2] = -1;
            *depth_map_ptr = -1;
        }
        else {
            float cost_pre = c[min_idx - 1];
            float cost_post = c[min_idx + 1];
            float a = cost_pre + cost_post - 2.0f * min_cost;
            float b = cost_post - cost_pre;
            float subpixel_idx = min_idx - b/(2.0f * a);
            float z = bf / subpixel_idx;

            point_cloud_ptr[0] = z*(p_u - cx)/fx;
            point_cloud_ptr[1] = z*(p_v - cy)/fy;
            point_cloud_ptr[2] = z;
            *depth_map_ptr = z;
        }
    }
}

void Postprocessing(float* cost, size_t height, size_t width, int num_disparity,
                    float bf, float fx, float fy, float cx, float cy,
                    float* point_cloud, size_t point_cloud_step,
                    float* depth_map, size_t depth_map_step) {
    dim3 block(num_disparity);
    dim3 grid(width, height);

    PostprocessingKernel
    <<<grid, block>>>
    (cost, height, width, num_disparity, bf, fx, fy, cx, cy,
     point_cloud, point_cloud_step,
     depth_map, depth_map_step);

    cudaDeviceSynchronize();
}

__global__ void WinnerTakesAllDisparityKernel(float* cost, size_t height, size_t width, int num_disparity,
                                              unsigned char* dis_mat, size_t dis_step) {
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int d = threadIdx.x;

    unsigned char* dis_mat_ptr = dis_mat + (y * dis_step + x);

    __shared__ float c_min[128];
    __shared__ int c_idx[128];
    c_min[d] = cost[INDEX(y, x, d, width, num_disparity)];
    c_idx[d] = d;
    __syncthreads();
    for(int i = num_disparity/2; i > 0; i /= 2) {
        if(d < i && d + i < num_disparity && c_min[d + i] < c_min[d]) {
            c_min[d] = c_min[d + i];
            c_idx[d] = c_idx[d + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        float min_cost = c_min[0];
        int min_idx = c_idx[0];

        if(min_cost <= 0)
            *dis_mat_ptr = 0;
        else
            *dis_mat_ptr = min_idx;
    }
}

void WinnerTakesAllDisparity(float* cost, size_t height, size_t width, int num_disparity,
                             unsigned char* dis_mat, size_t dis_step) {
    dim3 block(num_disparity);
    dim3 grid(width, height);
    WinnerTakesAllDisparityKernel
            <<<grid, block>>>(cost, height, width, num_disparity, dis_mat, dis_step);
    cudaDeviceSynchronize();
}
#undef INDEX
