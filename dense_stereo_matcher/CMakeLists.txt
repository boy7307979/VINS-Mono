cmake_minimum_required(VERSION 3.1)
project(dense_stereo_matcher)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
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

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED
NO_MODULE # should be optional, tells CMake to use config mode
PATHS /usr/local # look here
NO_DEFAULT_PATH) # and don't look anywhere else

message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

include_directories(${catkin_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    #    -std=c++11 -O3 -gencode arch=compute_30,code=sm_30  # T440s
    #    -std=c++11 -O3 -gencode arch=compute_53,code=sm_53  # tx1
    -std=c++11 -O3 -gencode arch=compute_61,code=sm_61  # gtx 980 Ti / 1080
    )


catkin_package(
CATKIN_DEPENDS systrace
)

cuda_add_executable(dense_stereo_matcher
    src/dense_stereo_matcher_node.cpp
    src/stereo_matcher.cpp
    src/sgm.cu
    src/sgm.cpp
    )


target_link_libraries(dense_stereo_matcher ${catkin_LIBRARIES} ${OpenCV_LIBS} tbb) 


