cmake_minimum_required(VERSION 3.4)
project(ceo_mlcpp)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -fexceptions -g -ggdb") #fexceptions for iVox
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  image_geometry
  pcl_ros
  pcl_conversions
  cv_bridge
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

if(OpenCV_FOUND)
  message(WARNING "OpenCV Found: " ${OpenCV_VERSION})
endif()
if(PCL_FOUND)
  message(WARNING "PCL Found: " ${PCL_VERSION})
endif()
if(OPENMP_FOUND)
  message(WARNING "OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if(EIGEN_FOUND)
  message(WARNING "Eigen Found: " ${EIGEN3_INCLUDE_DIR})
endif()

catkin_package()

include_directories(
  include
  include/ivox3d
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)


#### for iVox, tbb
set(CUSTOM_TBB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/ivox3d/tbb2018_20170726oss")
set(TBB2018_INCLUDE_DIR "${CUSTOM_TBB_DIR}/include")
set(TBB2018_LIBRARY_DIR "${CUSTOM_TBB_DIR}/lib/intel64/gcc4.7")
include_directories(${TBB2018_INCLUDE_DIR})
link_directories(${TBB2018_LIBRARY_DIR})


#### main
add_executable(${PROJECT_NAME}_node src/main_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_LIBS} ${PCL_LIBRARIES} ${OpenMP_LIBS} tbb glog
  pcl_common
  pcl_registration
  pcl_filters
  pcl_sample_consensus
  pcl_octree
  pcl_io
)