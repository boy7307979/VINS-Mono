cmake_minimum_required(VERSION 2.8.3)
project(dense_stereo_matcher)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    systrace
    )

find_package(OpenCV REQUIRED
NO_MODULE # should be optional, tells CMake to use config mode
PATHS /usr/local # look here
NO_DEFAULT_PATH) # and don't look anywhere else

message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
CATKIN_DEPENDS systrace
)

add_executable(dense_stereo_matcher
    src/dense_stereo_matcher_node.cpp
    src/stereo_matcher.cpp
    )


target_link_libraries(dense_stereo_matcher ${catkin_LIBRARIES} ${OpenCV_LIBS}) 


