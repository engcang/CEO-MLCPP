cmake_minimum_required(VERSION 3.4)
project(ceo_mlcpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  tf_conversions
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

if(OpenCV_FOUND)
  message(WARNING "OpenCV Found: " ${OpenCV_VERSION})
endif()
if(PCL_FOUND)
  message(WARNING "PCL Found: " ${PCL_VERSION})
endif()
if(EIGEN_FOUND)
  message(WARNING "Eigen Found: " ${EIGEN3_INCLUDE_DIR})
endif()

catkin_package()

include_directories(
  include
  include/ikd-Tree
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)


########## Build
#### for ikd-Tree
add_library(ikdtree include/ikd-Tree/ikd_Tree.cpp)

#### main
add_executable(${PROJECT_NAME}_node src/main_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_LIBS} ${PCL_LIBRARIES} ikdtree
  pcl_common
  pcl_registration
  pcl_filters
  pcl_sample_consensus
  pcl_octree
  pcl_io
)

#### generating .pcd
add_executable(pcd_generating_node src/pcd_generating_node.cpp)
target_link_libraries(pcd_generating_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_LIBS} ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES}
  pcl_common
  pcl_registration
  pcl_filters
  pcl_sample_consensus
  pcl_octree
  pcl_io
)