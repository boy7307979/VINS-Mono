#pragma once
#include <opencv2/opencv.hpp>

class StereoMatcher {
public:
    StereoMatcher() {}
    ~StereoMatcher() {}

    void ReadParameters(const std::string& filename);
    void operator()(const cv::Mat& img_l, const cv::Mat& img_r);
    cv::Size mImageSize;
    cv::Mat mKl, mKr;
    cv::Mat mDl, mDr;
    cv::Mat mTrl, mRrl, mtrl;
    cv::Mat mR1, mR2, mP1, mP2, mQ;
    cv::Mat mM1l, mM2l, mM1r, mM2r;
    cv::cuda::GpuMat mcM1l, mcM2l, mcM1r, mcM2r;

    cv::Ptr<cv::cuda::StereoBM> mpStereoBM;
};
