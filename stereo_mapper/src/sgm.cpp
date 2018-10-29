#include "sgm.h"
#include "sgm_impl.h"
#include "tic_toc.h"
#include <ros/ros.h>

StereoSGM::StereoSGM() {}

StereoSGM::~StereoSGM() {}

void StereoSGM::ReadParameters(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    int width, height;
    width = fs["image_width"];
    height = fs["image_height"];
    cv::Size image_size(width, height);

    std::vector<double> intrinsic, distortion;
    fs["intrinsics_L"] >> intrinsic;
    fs["distortion_coefficients_L"] >> distortion;

    cv::Mat Kl = cv::Mat::eye(3, 3, CV_64F), Dl;
    Kl.at<double>(0, 0) = intrinsic[0];
    Kl.at<double>(1, 1) = intrinsic[1];
    Kl.at<double>(0, 2) = intrinsic[2];
    Kl.at<double>(1, 2) = intrinsic[3];
    Dl = (cv::Mat_<double>(4, 1) << distortion[0], distortion[1], distortion[2], distortion[3]);

    cv::Mat Kr = cv::Mat::eye(3, 3, CV_64F), Dr;
    fs["intrinsics_R"] >> intrinsic;

    cv::Mat Tbl, Tbr;
    Sophus::SE3d Trl;
    if(!intrinsic.empty()) {
        mbIsStereo = true;
        fs["distortion_coefficients_R"] >> distortion;
        Kr.at<double>(0, 0) = intrinsic[0];
        Kr.at<double>(1, 1) = intrinsic[1];
        Kr.at<double>(0, 2) = intrinsic[2];
        Kr.at<double>(1, 2) = intrinsic[3];
        Dr = (cv::Mat_<double>(4, 1) << distortion[0], distortion[1], distortion[2], distortion[3]);

        fs["T_BCL"] >> Tbl;
        fs["T_BCR"] >> Tbr;
        cv::Mat tmp_Trl = Tbr.inv() * Tbl;
        Eigen::Matrix4d e_Trl;
        cv::cv2eigen(tmp_Trl, e_Trl);
        Eigen::Quaterniond unit_quaternion(e_Trl.block<3,3>(0,0));
        unit_quaternion.normalize();
        e_Trl.block<3,3>(0, 0) = unit_quaternion.toRotationMatrix();
        Trl = Sophus::SE3d(e_Trl);
    }
    else
        mbIsStereo = false;

    float baseline = fs["baseline"];
    mNumDisparity = fs["num_disparities"];
    mP1 = fs["P1"];
    mP2 = fs["P2"];
    mGtau = fs["Gtau"];
    mQ1 = fs["Q1"];
    mQ2 = fs["Q2"];

    int downsample = fs["downsample"];
    if(downsample)
        mbDownSample = true;
    else
        mbDownSample = false;

    ROS_INFO_STREAM("[StereoSGM]num disparity: " << mNumDisparity);
    ROS_INFO_STREAM("[StereoSGM]P1: " << mP1);
    ROS_INFO_STREAM("[StereoSGM]P2: " << mP2);
    ROS_INFO_STREAM("[StereoSGM]Gtau: " << mGtau);
    ROS_INFO_STREAM("[StereoSGM]Q1: " << mQ1);
    ROS_INFO_STREAM("[StereoSGM]Q2: " << mQ2);
    ROS_INFO_STREAM("[StereoSGM]downsample: " << mbDownSample);
    ROS_INFO_STREAM("[StereoSGM]Dl: " << Dl.t());
    ROS_INFO_STREAM("[StereoSGM]Dr: " << Dr.t());
    fs.release();

    if(mbIsStereo)
        InitIntrinsic(image_size, Kl, Dl, Kr, Dr, Trl, baseline);
    else
        InitIntrinsic(image_size, Kl, Dl, baseline);
}

void StereoSGM::InitIntrinsic(const cv::Size& image_size, const cv::Mat& Kl,
                              const cv::Mat& Dl, const cv::Mat& Kr, const cv::Mat& Dr,
                              const Sophus::SE3d& Trl, float baseline) {
    ROS_INFO_STREAM("[StereoSGM]Init stereo");
    mbIsStereo = true;
    mKl = Kl.clone();
    mKr = Kr.clone();
    mImageSize = image_size;
    if(mbDownSample) {
        mKl /= 2;
        mKl.at<double>(2, 2) = 1;
        mKr /= 2;
        mKr.at<double>(2, 2) = 1;
        mImageSize /= 2;
    }
    mImageArea = mImageSize.height * mImageSize.width;
    cv::Mat I3 = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(Kl, Dl, I3, mKl, mImageSize, CV_32F, mM1l, mM2l);
    cv::initUndistortRectifyMap(Kr, Dr, I3, mKr, mImageSize, CV_32F, mM1r, mM2r);
    mcM1l.upload(mM1l);
    mcM2l.upload(mM2l);
    mcM1r.upload(mM1r);
    mcM2r.upload(mM2r);
    mcSADCost.create(1, mImageArea * mNumDisparity, CV_32F);
    mcSGMCost.create(1, mImageArea * mNumDisparity, CV_32F);
    mcPointCloud.create(mImageSize, CV_32FC3);
    mcDepth.create(mImageSize, CV_32F);

    mTrl = Trl;
    mTlr = mTrl.inverse();
    mbf = baseline * mKl.at<double>(0, 0);

    ROS_INFO_STREAM("[StereoSGM]Kl:\n" << mKl);
    ROS_INFO_STREAM("[StereoSGM]Kr:\n" << mKr);
    ROS_INFO_STREAM("[StereoSGM]image size: " << mImageSize);
    ROS_INFO_STREAM("[StereoSGM]image area: " << mImageArea);
    ROS_INFO_STREAM("[StereoSGM]Trl:\n" << mTrl.matrix3x4());
    ROS_INFO_STREAM("[StereoSGM]bf: " << mbf);
}

void StereoSGM::InitIntrinsic(const cv::Size& image_size, const cv::Mat& K, const cv::Mat& D, float baseline) {
    ROS_INFO_STREAM("[StereoSGM]Init mono");
    mbIsStereo = false;
    mKl = K.clone();
    mImageSize = image_size;
    if(mbDownSample) {
        mKl /= 2;
        mKl.at<double>(2, 2) = 1;
        mImageSize /= 2;
    }
    mImageArea = mImageSize.height * mImageSize.width;
    cv::Mat I3 = cv::Mat::eye(3, 3, CV_32F);
    cv::initUndistortRectifyMap(K, D, I3, mKl, mImageSize, CV_32F, mM1l, mM2l);
    mcM1l.upload(mM1l);
    mcM2l.upload(mM2l);
    mcSADCost.create(1, mImageArea * mNumDisparity, CV_32F);
    mcSGMCost.create(1, mImageArea * mNumDisparity, CV_32F);
    mcPointCloud.create(mImageSize, CV_32FC3);
    mcDepth.create(mImageSize, CV_32F);
    mbf = baseline * mKl.at<double>(0, 0);

    ROS_INFO_STREAM("[StereoSGM]Kl:\n" << mKl);
    ROS_INFO_STREAM("[StereoSGM]image size: " << mImageSize);
    ROS_INFO_STREAM("[StereoSGM]image area: " << mImageArea);
    ROS_INFO_STREAM("[StereoSGM]bf: " << mbf);
}

void StereoSGM::InitReference(const cv::Mat& img_ref, const Sophus::SE3d& Tw_ref) {
    TicToc tic;
    mTw_ref = Tw_ref;
    mTref_w = mTw_ref.inverse();

    //    cv::cuda::GpuMat img_ref_; // something wrong, i don't know why
    //    img_ref_.upload(img_ref);
    //    img_ref_.convertTo(aaa, CV_32F); //???
    cv::Mat img_ref_;
    img_ref.convertTo(img_ref_, CV_32F); // FIXME??
    mcRefImg.upload(img_ref_);
    cv::cuda::remap(mcRefImg, mcRefImg, mcM1l, mcM2l, cv::INTER_LINEAR);
    mMeasCount = 0;
    if(mbDebug) {
        mcRefImg.download(mRefImg);
        mRefImg.convertTo(mRefImg, CV_8U);
        cv::imshow("ref_img", mRefImg);
        cv::waitKey(1);
    }
}

void StereoSGM::UpdateByMotion(const cv::Mat& img_cur, const Sophus::SE3d& Tw_cur) {
    ++mMeasCount;
    mTw_cur = Tw_cur;
    mTcur_w = Tw_cur.inverse();

    cv::Mat img_cur_;
    img_cur.convertTo(img_cur_, CV_32F);
    mcCurImg.upload(img_cur_);
    cv::cuda::remap(mcCurImg, mcCurImg, mcM1l, mcM2l, cv::INTER_LINEAR);

    if(mbDebug) {
        mcCurImg.download(mCurImg);
        mCurImg.convertTo(mCurImg, CV_8U);
        cv::imshow("cur_img", mCurImg);
        cv::waitKey(1);
    }

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kl(mKl.ptr<double>());
    Eigen::Matrix3d Rw_ref = mTw_ref.rotationMatrix(), Rcur_w = mTcur_w.rotationMatrix();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H = (Kl * Rcur_w * Rw_ref * Kl.inverse()).cast<float>();
    Eigen::Vector3f h = (Kl * Rcur_w * (mTw_ref.translation() - mTw_cur.translation())).cast<float>();
    TicToc tic;
    SADCalcCost(mMeasCount, mNumDisparity, mbf, H.data(), h.data(), (float*)mcRefImg.data, (float*)mcCurImg.data,
                mImageSize.height, mImageSize.width, mcCurImg.step, (float*)mcSADCost.data);
    ROS_INFO_STREAM("[StereoSGM]SADCalcCost: " << tic.toc() << " (ms)");
}

void StereoSGM::UpdateByMotionR(const cv::Mat& img_cur, const Sophus::SE3d& Tw_cur) {
    ++mMeasCount;
    mTw_cur = Tw_cur * mTlr;
    mTcur_w = mTrl * Tw_cur.inverse();

    cv::Mat img_cur_;
    img_cur.convertTo(img_cur_, CV_32F);
    mcCurImg.upload(img_cur_);
    cv::cuda::remap(mcCurImg, mcCurImg, mcM1r, mcM2r, cv::INTER_LINEAR);

    if(mbDebug) {
        mcCurImg.download(mCurImg);
        mCurImg.convertTo(mCurImg, CV_8U);
        cv::imshow("cur_img", mCurImg);
        cv::waitKey(1);
    }

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kl(mKl.ptr<double>());
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kr(mKr.ptr<double>());
    Eigen::Matrix3d Rw_ref = mTw_ref.rotationMatrix(), Rcur_w = mTcur_w.rotationMatrix();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H = (Kr * Rcur_w * Rw_ref * Kl.inverse()).cast<float>();
    Eigen::Vector3f h = (Kr * Rcur_w * (mTw_ref.translation() - mTw_cur.translation())).cast<float>();
    TicToc tic;
    SADCalcCost(mMeasCount, mNumDisparity, mbf, H.data(), h.data(), (float*)mcRefImg.data, (float*)mcCurImg.data,
                mImageSize.height, mImageSize.width, mcCurImg.step, (float*)mcSADCost.data);
    ROS_INFO_STREAM("[StereoSGM]SADCalcCost: " << tic.toc() << " (ms)");
}

void StereoSGM::GetResult(cv::Mat* unimg_ref, cv::Mat* point_cloud,
                          cv::Mat* depth_map) {
    if(!unimg_ref && !point_cloud && !depth_map)
        return;
    mcSGMCost.setTo(cv::Scalar_<float>(0.0));
    TicToc tic;
    SGM4PathCalcCost(mP1, mP2, mGtau, mQ1, mQ2, mNumDisparity,
                     mImageSize.height, mImageSize.width,
                     (float*)mcSADCost.data, (float*)mcSGMCost.data);
    ROS_INFO_STREAM("[StereoSGM]SGMCalcCost: " << tic.toc() << " (ms)");

    if(point_cloud || depth_map) {
        tic.tic();
        Postprocessing((float*)mcSGMCost.data, mImageSize.height, mImageSize.width, mNumDisparity,
                       mbf, mKl.at<double>(0, 0), mKl.at<double>(1, 1), mKl.at<double>(0, 2),
                       mKl.at<double>(1, 2), (float*)mcPointCloud.data, mcPointCloud.step,
                       (float*)mcDepth.data, mcDepth.step);
        ROS_INFO_STREAM("[StereoSGM]Postprocessing: " << tic.toc() << " (ms)");
    }

    if(unimg_ref) {
        mcRefImg.download(*unimg_ref);
        unimg_ref->convertTo(*unimg_ref, CV_8U);
    }

    if(point_cloud)
        mcPointCloud.download(*point_cloud);

    if(depth_map)
        mcDepth.download(*depth_map);
}
