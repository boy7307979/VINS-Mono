#pragma once
#include <opencv2/opencv.hpp>

class StereoMatcher {
public:
    StereoMatcher() {}
    ~StereoMatcher() {}

    void ReadParameters(const std::string& filename);
    void compute(const cv::Mat& img_l, const cv::Mat& img_r,
                    cv::Mat& img_rect_l, cv::Mat& disparity,
                 cv::Mat& point_cloud);

    cv::Size mImageSize;
    cv::Mat mKl, mKr;
    cv::Mat mDl, mDr;
    cv::Mat mTrl, mRrl, mtrl;
    cv::Mat mR1, mR2, mP1, mP2, mQ;
    cv::Mat mM1l, mM2l, mM1r, mM2r;

    cv::cuda::GpuMat mcM1l, mcM2l, mcM1r, mcM2r;
    cv::cuda::GpuMat gpu_img_l, gpu_img_r;
    cv::cuda::GpuMat gpu_rect_img_l, gpu_rect_img_r;
    cv::cuda::GpuMat gpu_disparity;
    cv::cuda::GpuMat gpu_point_cloud;
    cv::Ptr<cv::cuda::StereoBM> mpStereoBM;
};
