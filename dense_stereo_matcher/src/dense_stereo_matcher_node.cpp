#include <ros/ros.h>
#include <queue>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cv_bridge/cv_bridge.h>
#include "stereo_matcher.h"

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
        fs.release();
        mStereoMatcher.ReadParameters(filename);
    }

    void ImageCallback(const sensor_msgs::ImageConstPtr& img_l_msg,
                       const sensor_msgs::ImageConstPtr& img_r_msg) {
        mMutex.lock();
        mImageBuffer.emplace(img_l_msg, img_r_msg);
        mMutex.unlock();
        mCV.notify_one();
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
            mStereoMatcher(left_img, right_img);
        }
    }

    std::string image_l_topic, image_r_topic;
    ImageBuffer mImageBuffer;
    StereoMatcher mStereoMatcher;
    std::mutex mMutex;
    std::condition_variable mCV;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "dense_stereo_matcher");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    Node node;
    node.ReadParameters(nh);

    message_filters::Subscriber<Image> sub_image_l(nh, node.image_l_topic, 100);
    message_filters::Subscriber<Image> sub_image_r(nh, node.image_r_topic, 100);
    TimeSynchronizer<Image, Image> sync(sub_image_l, sub_image_r, 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    std::thread main_thread(&Node::Process, &node);

    ros::spin();
    return 0;
}
