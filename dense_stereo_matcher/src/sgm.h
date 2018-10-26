#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

class StereoSGM {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoSGM();
    ~StereoSGM();

    void ReadParameters(const std::string& filename);
    void InitIntrinsic(const cv::Size& image_size, const cv::Mat& Kl, const cv::Mat& Dl,
                       const cv::Mat& Kr, const cv::Mat& Dr, const Sophus::SE3d& Trl, float baseline);
    void InitIntrinsic(const cv::Size& image_size,const cv::Mat& K, const cv::Mat& D, float baseline);
    void InitReference(const cv::Mat& img_ref, const Sophus::SE3d& Tw_ref);
    void UpdateByMotion(const cv::Mat& img_cur, const Sophus::SE3d& Tw_cur);
    void UpdateByMotionR(const cv::Mat& img_cur, const Sophus::SE3d& Tw_cur);
    void ShowDisparity();
private:
    // default setting
    bool mbIsStereo = false;
    bool mbDownSample = true;
    int mMeasCount = 0;

    // sgm parameters setting
    int mNumDisparity = 64;
    float mP1;
    float mP2;
    float mGtau;
    float mQ1;
    float mQ2;

    // camera coefficient
    int mImageArea;
    float mbf;
    cv::Size mImageSize;
    cv::Mat mKl, mKr;
    cv::Mat mM1l, mM2l, mM1r, mM2r;
    Sophus::SE3d mTrl, mTlr, mTw_ref, mTref_w, mTw_cur, mTcur_w;

    // gpu setting
    cv::cuda::GpuMat mcM1l, mcM2l, mcM1r, mcM2r;
    cv::cuda::GpuMat mcRefImg, mcCurImg; // undistort image
    cv::cuda::GpuMat mcSADCost, mcSGMCost;
    cv::cuda::GpuMat mcDepth;

    // debug info
    bool mbDebug = true;
    cv::Mat mRefImg, mCurImg;
};

using StereoSGMPtr = std::shared_ptr<StereoSGM>;
using StereoSGMConstPtr = std::shared_ptr<const StereoSGM>;
