#include "sgm.h"
#include "sgm_impl.h"

StereoSGM::StereoSGM() {}

StereoSGM::~StereoSGM() {}

void StereoSGM::InitIntrinsic(const cv::Size& image_size, const cv::Mat& Kl,
                              const cv::Mat& Dl, const cv::Mat& Kr, const cv::Mat& Dr,
                              const Sophus::SE3d& Trl) {
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
    mcDepth.create(mImageSize, CV_32F);

    mTrl = Trl;
    mTlr = mTrl.inverse();
}

void StereoSGM::InitIntrinsic(const cv::Size& image_size, const cv::Mat& K, const cv::Mat& D, float baseline) {
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
    mcDepth.create(mImageSize, CV_32F);
    mbf = baseline * mKl.at<double>(0, 0);
}

void StereoSGM::InitReference(const cv::Mat& img_ref, const Sophus::SE3d& Tw_ref) {
    mTw_ref = Tw_ref;
    mTref_w = mTw_ref.inverse();

    cv::cuda::GpuMat c_img_ref;
    c_img_ref.upload(img_ref);
    c_img_ref.convertTo(c_img_ref, CV_32F);
    cv::cuda::remap(c_img_ref, mcRefImg, mcM1l, mcM2l, cv::INTER_LINEAR);
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

    cv::cuda::GpuMat c_img_cur;
    c_img_cur.upload(img_cur);
    c_img_cur.convertTo(c_img_cur, CV_32F);
    cv::cuda::remap(c_img_cur, mcCurImg, mcM1l, mcM2l, cv::INTER_LINEAR);

    if(mbDebug) {
        mcCurImg.download(mCurImg);
        mCurImg.convertTo(mCurImg, CV_8U);
        cv::imshow("cur_img", mCurImg);
        cv::waitKey(1);
    }

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kl(mKl.ptr<double>());
    Eigen::Matrix3d Rref_w = mTref_w.rotationMatrix(), Rw_cur = mTw_cur.rotationMatrix();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H = (Kl * Rref_w * Rw_cur * Kl.inverse()).cast<float>();
    Eigen::Vector3f h = (Kl * Rref_w * (mTw_ref.translation() - mTw_cur.translation())).cast<float>();

    SADCalcCost(mMeasCount, mNumDisparity, mbf, H.data(), h.data(), (float*)mcRefImg.data, (float*)mcCurImg.data,
                mImageSize.height, mImageSize.width, mcCurImg.step, (float*)mcSADCost.data);
}

void StereoSGM::UpdateByMotionR(const cv::Mat& img_cur, const Sophus::SE3d& Tw_cur) {
    ++mMeasCount;
    mTw_cur = Tw_cur * mTlr;

    cv::cuda::GpuMat c_img_cur;
    c_img_cur.upload(img_cur);
    c_img_cur.convertTo(c_img_cur, CV_32F);
    cv::cuda::remap(c_img_cur, mcCurImg, mcM1l, mcM2l, cv::INTER_LINEAR);

    if(mbDebug) {
        mcCurImg.download(mCurImg);
        mCurImg.convertTo(mCurImg, CV_8U);
        cv::imshow("cur_img", mCurImg);
        cv::waitKey(1);
    }

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kl(mKl.ptr<double>());
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Kr(mKr.ptr<double>());
    Eigen::Matrix3d Rref_w = mTref_w.rotationMatrix(), Rw_cur = mTw_cur.rotationMatrix();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H = (Kl * Rref_w * Rw_cur * Kr.inverse()).cast<float>();
    Eigen::Vector3f h = (Kl * Rref_w * (mTw_ref.translation() - mTw_cur.translation())).cast<float>();

    SADCalcCost(mMeasCount, mNumDisparity, mbf, H.data(), h.data(), (float*)mcRefImg.data, (float*)mcCurImg.data,
                mImageSize.height, mImageSize.width, mcCurImg.step, (float*)mcSADCost.data);
}

void StereoSGM::ShowDisparity() {
    SGM4PathCalcCost(mP1, mP2, mGtau, mQ1, mQ2, mNumDisparity,
                     mImageSize.height, mImageSize.width,
                     (float*)mcSADCost.data, (float*)mcSGMCost.data);
    Postprocessing((float*)mcSGMCost.data, mImageSize.height, mImageSize.width, mNumDisparity,
                   (float*)mcDepth.data, mcDepth.step);
    cv::Mat disparity;
    mcDepth.download(disparity);
    disparity.convertTo(disparity, CV_8U);
    cv::imshow("disparity", disparity);
    cv::waitKey(1);
}
