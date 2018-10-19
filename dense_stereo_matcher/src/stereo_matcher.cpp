#include "stereo_matcher.h"
#include "tic_toc.h"
#include <ros/ros.h>
#include <chrono>
#include <thread>

void StereoMatcher::ReadParameters(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    int width, height;
    width = fs["image_width"];
    height = fs["image_height"];
    mImageSize.width = width;
    mImageSize.height = height;
    cv::Mat Tbl, Tbr;
    fs["T_BCL"] >> Tbl;
    fs["T_BCR"] >> Tbr;
    mTrl = Tbr.inv() * Tbl;
    mTrl.colRange(0, 3).rowRange(0, 3).copyTo(mRrl);
    mTrl.col(3).rowRange(0, 3).copyTo(mtrl);

    std::vector<double> intrinsic, distortion;
    fs["intrinsics_L"] >> intrinsic;
    fs["distortion_coefficients_L"] >> distortion;
    mKl = cv::Mat::eye(3, 3, CV_64F);
    mKl.at<double>(0, 0) = intrinsic[0];
    mKl.at<double>(1, 1) = intrinsic[1];
    mKl.at<double>(0, 2) = intrinsic[2];
    mKl.at<double>(1, 2) = intrinsic[3];
    mDl = (cv::Mat_<double>(4, 1) << distortion[0], distortion[1], distortion[2], distortion[3]);

    fs["intrinsics_R"] >> intrinsic;
    fs["distortion_coefficients_R"] >> distortion;
    mKr = cv::Mat::eye(3, 3, CV_64F);
    mKr.at<double>(0, 0) = intrinsic[0];
    mKr.at<double>(1, 1) = intrinsic[1];
    mKr.at<double>(0, 2) = intrinsic[2];
    mKr.at<double>(1, 2) = intrinsic[3];
    mDr = (cv::Mat_<double>(4, 1) << distortion[0], distortion[1], distortion[2], distortion[3]);

    int numDisparities, blockSize;
    numDisparities = fs["numDisparities"];
    blockSize = fs["blockSize"];
    fs.release();

    cv::stereoRectify(mKl, mDl, mKr, mDr, mImageSize, mRrl, mtrl, mR1, mR2, mP1, mP2, mQ,
                      cv::CALIB_ZERO_DISPARITY);
    cv::initUndistortRectifyMap(mKl, mDl, mR1, mP1, mImageSize, CV_32F, mM1l, mM2l);
    cv::initUndistortRectifyMap(mKr, mDr, mR2, mP2, mImageSize, CV_32F, mM1r, mM2r);

    mcM1l.upload(mM1l);
    mcM2l.upload(mM2l);
    mcM1r.upload(mM1r);
    mcM2r.upload(mM2r);

    mQ.convertTo(mQ, CV_32F);
    mpStereoBM = cv::cuda::createStereoBM(numDisparities, blockSize);

    ROS_INFO_STREAM("disparity");
    for(int i = 1; i < 64; ++i) {
        ROS_INFO_STREAM("i -> " << -mP2.at<double>(0, 3)/i << " (m)");
    }
}

void StereoMatcher::compute(const cv::Mat& img_l, const cv::Mat& img_r,
                            cv::Mat& img_rect_l, cv::Mat& disparity,
                            cv::Mat& point_cloud) {
    TicToc tic;
    gpu_img_l.upload(img_l);
    gpu_img_r.upload(img_r);
    cv::cuda::remap(gpu_img_l, gpu_rect_img_l, mcM1l, mcM2l, cv::INTER_LINEAR);
    cv::cuda::remap(gpu_img_r, gpu_rect_img_r, mcM1r, mcM2r, cv::INTER_LINEAR);
    mpStereoBM->compute(gpu_rect_img_l, gpu_rect_img_r, gpu_disparity);

    cv::cuda::reprojectImageTo3D(gpu_disparity, gpu_point_cloud, mQ, 3);
    gpu_disparity.download(disparity);
    gpu_rect_img_l.download(img_rect_l);
    gpu_point_cloud.download(point_cloud);
//    double min_value, max_value;
//    cv::minMaxIdx(disparity, &min_value, &max_value);
//    cv::Mat show_disparity;
//    disparity.convertTo(show_disparity, CV_8U, 255.0/max_value);
//    cv::imshow("disparity", show_disparity);
//    cv::waitKey(1);
}
