#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <memory>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "sgm.h"

using namespace std;
using namespace message_filters;
using namespace sensor_msgs;
using namespace geometry_msgs;

class Node {
public:
    void ReadParameters(const ros::NodeHandle& nh) {
        std::string filename;
        nh.getParam("config_file", filename);
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        fs["image_L_topic"] >> img_l_topic;
        fs["image_R_topic"] >> img_r_topic;
        fs["ref_pose_topic"] >> ref_pose_topic;
        fs["cur_pose_topic"] >> cur_pose_topic;
        fs.release();
        mStereoSGM.ReadParameters(filename);
    }

    void InitPub(ros::NodeHandle& nh) {
        mPubPointCloud = nh.advertise<sensor_msgs::PointCloud2>("point_cloud2", 1000);
        mPubImg = nh.advertise<sensor_msgs::Image>("rgb/image_raw", 1000);
        mPubImgInfo = nh.advertise<sensor_msgs::CameraInfo>("rgb/image_info", 1000);
        mPubDepthImg = nh.advertise<sensor_msgs::Image>("depth/image_raw", 1000);
        mPubDepthImgInfo = nh.advertise<sensor_msgs::CameraInfo>("depth/image_info", 1000);
    }

    void InitSub(ros::NodeHandle& nh) {
        mSubImgL.subscribe(nh, img_l_topic, 100);
        mSubImgR.subscribe(nh, img_r_topic, 100);
        mSubCurPose.subscribe(nh, cur_pose_topic, 100);
        mpSyncPolocy = std::make_shared<
                TimeSynchronizer<Image, Image, PoseStamped>>
                (mSubImgL, mSubImgR, mSubCurPose, 100);
        mpSyncPolocy->registerCallback(boost::bind(&Node::StereoCallback, this, _1, _2, _3));
    }

    void StereoCallback(const ImageConstPtr& img_l_ptr, const ImageConstPtr& img_r_ptr,
                        const PoseStampedConstPtr& cur_pose_ptr) {

        cv::Mat img_l = cv_bridge::toCvCopy(img_l_ptr, "mono8")->image;
        cv::Mat img_r = cv_bridge::toCvCopy(img_r_ptr, "mono8")->image;
        Eigen::Quaterniond q_wl(cur_pose_ptr->pose.orientation.w,
                                cur_pose_ptr->pose.orientation.x,
                                cur_pose_ptr->pose.orientation.y,
                                cur_pose_ptr->pose.orientation.z);
        Eigen::Vector3d t_wl(cur_pose_ptr->pose.position.x,
                             cur_pose_ptr->pose.position.y,
                             cur_pose_ptr->pose.position.z);
        {
            // camera frame
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(t_wl.x(),
                                            t_wl.y(),
                                            t_wl.z()));
            q.setW(q_wl.w());
            q.setX(q_wl.x());
            q.setY(q_wl.y());
            q.setZ(q_wl.z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, cur_pose_ptr->header.stamp, "world", "ref_frame"));
            mKeyHeader = cur_pose_ptr->header;
        }

        Sophus::SE3d Twl(q_wl, t_wl);
        mStereoSGM.InitReference(img_l, Twl);
        mStereoSGM.UpdateByMotionR(img_r, Twl);

        mStereoSGM.GetResult(&mImgRectL, &mPointCloud, &mDepthMap);
        Publish();
//        double min, max;
//        cv::minMaxIdx(mDepthMap, &min, &max);
//        mDepthMap.convertTo(mDepthMap, CV_8U, 255.0/max);
//        cv::imshow("ddd", mDepthMap);
//        cv::waitKey(1);
    }

    void Publish() {
        // part1 point cloud
        sensor_msgs::PointCloud2Ptr points(new sensor_msgs::PointCloud2);
        points->header = mKeyHeader;
        points->header.frame_id = "ref_frame";

        points->height = mPointCloud.rows;
        points->width = mPointCloud.cols;
        points->fields.resize(4);
        points->fields[0].name = "x";
        points->fields[0].offset = 0;
        points->fields[0].count = 1;
        points->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        points->fields[1].name = "y";
        points->fields[1].offset = 4;
        points->fields[1].count = 1;
        points->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        points->fields[2].name = "z";
        points->fields[2].offset = 8;
        points->fields[2].count = 1;
        points->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        points->fields[3].name = "rgb";
        points->fields[3].offset = 12;
        points->fields[3].count = 1;
        points->fields[3].datatype = sensor_msgs::PointField::UINT32;
        points->point_step = 16;
        points->row_step = points->point_step * points->width;
        points->data.resize(points->row_step * points->height);
        points->is_dense = false; // there may be invalid points

        float bad_point = std::numeric_limits<float>::quiet_NaN();
        const float* point_cloud_ptr = mPointCloud.ptr<const float>();
        const uchar* rect_img_ptr = mImgRectL.ptr<const uchar>();
        for(int i = 0, n = mImgRectL.cols * mImgRectL.rows; i < n;
            ++i, point_cloud_ptr+=3, ++rect_img_ptr) {
            float z = point_cloud_ptr[2];
            if(z == -1 || std::isnan(z) || std::isinf(z)) {
                size_t index = i * points->point_step;
                memcpy(&points->data[index + 0], &bad_point, sizeof(float));
                memcpy(&points->data[index + 4], &bad_point, sizeof(float));
                memcpy(&points->data[index + 8], &bad_point, sizeof(float));
                memcpy(&points->data[index + 12], &bad_point, sizeof(float));
            }
            else {
                int index = i * points->point_step;
                uint8_t intensity = *rect_img_ptr;
                uint32_t rgb = (intensity << 16) | (intensity << 8) | intensity;
                memcpy(&points->data[index + 0], point_cloud_ptr, sizeof(float)*3);
                memcpy(&points->data[index + 12], &rgb, sizeof(uint32_t));
            }
        }

        mPubPointCloud.publish(points);

        // part 2 publish img
        cv_bridge::CvImage output_img_msg;
        output_img_msg.header = mKeyHeader;
        output_img_msg.header.frame_id = "camera";
        output_img_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        output_img_msg.image = mDepthMap.clone();
        mPubDepthImg.publish(output_img_msg.toImageMsg());

        output_img_msg.encoding = sensor_msgs::image_encodings::MONO8;
        output_img_msg.image = mImgRectL.clone();
        mPubImg.publish(output_img_msg.toImageMsg());

        // part 3 publish camera coefficient
        sensor_msgs::CameraInfo camera_info;
        camera_info.header = mKeyHeader;
        const cv::Mat& Kl = mStereoSGM.Kl();
        double fx = Kl.at<double>(0, 0);
        double fy = Kl.at<double>(1, 1);
        double cx = Kl.at<double>(0, 2);
        double cy = Kl.at<double>(1, 2);
        camera_info.P[0] = fx;
        camera_info.P[2] = cx;
        camera_info.P[5] = fy;
        camera_info.P[6] = cy;

        const cv::Size& image_size = mStereoSGM.ImageSize();
        camera_info.width = image_size.width;
        camera_info.height = image_size.height;
        mPubImgInfo.publish(camera_info);
        mPubDepthImgInfo.publish(camera_info);
    }

private:
    std::string img_l_topic, img_r_topic;
    std::string ref_pose_topic, cur_pose_topic;
    ros::Publisher mPubPointCloud, mPubImg, mPubImgInfo,
                   mPubDepthImg, mPubDepthImgInfo;
    message_filters::Subscriber<sensor_msgs::Image> mSubImgL, mSubImgR;
    message_filters::Subscriber<geometry_msgs::PoseStamped> mSubCurPose;
    std::shared_ptr<TimeSynchronizer<Image, Image, PoseStamped>> mpSyncPolocy;

    std_msgs::Header mKeyHeader;
    cv::Mat mImgRectL, mPointCloud, mDepthMap;

    StereoSGM mStereoSGM;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "stereo_mapper");
    ros::NodeHandle nh("~");

    Node node;
    node.ReadParameters(nh);
    node.InitPub(nh);
    node.InitSub(nh);

    ros::spin();
    return 0;
}
