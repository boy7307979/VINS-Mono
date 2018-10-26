#include <ros/ros.h>
#include <queue>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include "sgm.h"

using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using ImageBuffer = queue<pair<ImageConstPtr, ImageConstPtr>>;
    Node() {}
    ~Node() {}

    void ReadParameters(const ros::NodeHandle& nh) {
        std::string filename;
        nh.getParam("config_file", filename);
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        fs["image_L_topic"] >> image_l_topic;
        fs["image_R_topic"] >> image_r_topic;
        mStereoSGM.ReadParameters(filename);
        fs.release();
    }

    void RegisterPub(ros::NodeHandle& nh) {
        pub_point_cloud2 = nh.advertise<sensor_msgs::PointCloud2>("point_cloud", 1000);
    }

    void ImageCallback(const sensor_msgs::ImageConstPtr& img_l_msg,
                       const sensor_msgs::ImageConstPtr& img_r_msg) {
        mMutex.lock();
        mImageBuffer.emplace(img_l_msg, img_r_msg);
        mMutex.unlock();
        mCV.notify_one();
    }

    void SendCloud(const sensor_msgs::ImageConstPtr& msg,
                   const cv::Mat& rect_img, const cv::Mat& point_cloud) {
        sensor_msgs::PointCloud2Ptr points(new sensor_msgs::PointCloud2);
        points->header = msg->header;
        points->header.frame_id = "world";

        points->height = rect_img.rows;
        points->width = rect_img.cols;
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
        const float* point_cloud_ptr = point_cloud.ptr<const float>();
        const uchar* rect_img_ptr = rect_img.ptr<const uchar>();
        for(int i = 0, n = rect_img.rows * rect_img.cols; i < n;
            ++i, point_cloud_ptr+=3, ++rect_img_ptr) {
            if(std::isinf(*point_cloud_ptr) || std::isnan(*point_cloud_ptr)) {
                int index = i * points->point_step;
                memcpy(&points->data[index + 0], &bad_point, sizeof(float));
                memcpy(&points->data[index + 4], &bad_point, sizeof(float));
                memcpy(&points->data[index + 8], &bad_point, sizeof(float));
                memcpy(&points->data[index + 12], &bad_point, sizeof(uint32_t));
            }
            else {
                int index = i * points->point_step;
                uint8_t intensity = *rect_img_ptr;
                uint32_t rgb = (intensity << 16) | (intensity << 8) | intensity;
                memcpy(&points->data[index + 0], point_cloud_ptr, sizeof(float)*3);
                memcpy(&points->data[index + 12], &rgb, sizeof(uint32_t));
            }
        }
        pub_point_cloud2.publish(points);
    }

    void Process() {
        while(1) {
            std::unique_lock<std::mutex> lock(mMutex);
            mCV.wait(lock, [&]{
                return !mImageBuffer.empty();
            });
            pair<ImageConstPtr, ImageConstPtr> image_pair;
            while(!mImageBuffer.empty()) {
                image_pair = mImageBuffer.front();
                mImageBuffer.pop();
            }
            lock.unlock();

            cv::Mat left_img = cv_bridge::toCvCopy(image_pair.first, "mono8")->image;
            cv::Mat right_img = cv_bridge::toCvCopy(image_pair.second, "mono8")->image;
            cv::Mat disparity, point_cloud, left_rect_img;
            Sophus::SE3d T;
            mStereoSGM.InitReference(left_img, T);
            mStereoSGM.UpdateByMotionR(right_img, T);
            mStereoSGM.ShowDisparity();
        }
    }

    std::string image_l_topic, image_r_topic;
    ImageBuffer mImageBuffer;

    std::mutex mMutex;
    std::condition_variable mCV;
    ros::Publisher pub_point_cloud2;
    StereoSGM mStereoSGM;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "dense_stereo_matcher");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    Node node;
    node.ReadParameters(nh);
    node.RegisterPub(nh);

    message_filters::Subscriber<Image> sub_image_l(nh, node.image_l_topic, 100);
    message_filters::Subscriber<Image> sub_image_r(nh, node.image_r_topic, 100);
    TimeSynchronizer<Image, Image> sync(sub_image_l, sub_image_r, 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    std::thread main_thread(&Node::Process, &node);

    ros::spin();
    return 0;
}
