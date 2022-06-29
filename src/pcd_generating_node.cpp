///// C++
#include <string>
#include <math.h>
#include <mutex>
#include <chrono>
#include <utility> 
#include <vector> 

///// Eigen, Linear Algebra
#include <Eigen/Eigen> //whole Eigen library

///// ROS
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h> // to Quaternion_to_euler
#include <tf/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf/LinearMath/Matrix3x3.h> // to Quaternion_to_euler
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>

///// OctoMap
#include <octomap/OccupancyOcTreeBase.h>
#include <octomap/OcTree.h>
#include <octomap/octomap.h>
#include <octomap/math/Utils.h>

///// PCL
//defaults
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
//conversions
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
//voxel, filters, etc
#include <pcl/filters/passthrough.h>

using namespace std;

//////////////////////////////////////////////////////
#include <signal.h>
void signal_handler(sig_atomic_t s) {
  std::cout << "You pressed Ctrl + C, exiting" << std::endl;
  exit(1);
}
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<pcl::PointXYZ> cloud, std::string frame_id = "map")
{
  sensor_msgs::PointCloud2 cloud_ROS;
  pcl::toROSMsg(cloud, cloud_ROS);
  cloud_ROS.header.frame_id = frame_id;
  return cloud_ROS;
}
pcl::PointCloud<pcl::PointXYZ> cloudmsg2cloud(const sensor_msgs::PointCloud2 &cloudmsg)
{
  pcl::PointCloud<pcl::PointXYZ> cloudresult;
  pcl::fromROSMsg(cloudmsg,cloudresult);
  return cloudresult;
}

//////////////////////////////////////////////////////
shared_ptr<octomap::OcTree> m_octree = nullptr;
pcl::PointCloud<pcl::PointXYZ> m_pcd_out;
pair<double, pcl::PointCloud<pcl::PointXYZ>> m_pcl_input;
pair<double, Eigen::Matrix<double, 4, 4>> m_pose_input;
double m_pcd_voxel_size=0.5;
Eigen::Matrix<double, 4, 4> m_body_t_lidar = Eigen::Matrix<double, 4, 4>::Identity();
vector<double> m_lidar_tf;
ros::Publisher m_pcl_debug_pub;
mutex m_mutex_pcl, m_mutex_pose;


//////////////////////////////////////////////////////
void save_cb(const std_msgs::Empty::ConstPtr& msg){
  pcl::PointCloud<pcl::PointXYZ>::Ptr before_filter(new pcl::PointCloud<pcl::PointXYZ>());
  for (octomap::OcTree::iterator it=m_octree->begin(); it!=m_octree->end(); ++it){
    if(m_octree->isNodeOccupied(*it))
    {
      before_filter->push_back(pcl::PointXYZ(it.getCoordinate().x(), it.getCoordinate().y(), it.getCoordinate().z()));
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (before_filter);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (1.0, 999999.0);
  pass.filter (*filtered);
  pcl::io::savePCDFileASCII<pcl::PointXYZ> ("generated_pcd.pcd", *filtered);
  ROS_WARN("pcd saved");
}

void pcl_cb(const sensor_msgs::PointCloud2::ConstPtr& msg){
  {
    lock_guard<mutex> lock(m_mutex_pcl);
    m_pcl_input = make_pair(msg->header.stamp.toSec(), cloudmsg2cloud(*msg));
  }
}
void pose_cb(const nav_msgs::Odometry::ConstPtr& msg){
  geometry_msgs::Pose pose = msg->pose.pose;
  Eigen::Matrix<double, 4, 4> tmp_mat = Eigen::Matrix<double, 4, 4>::Identity();

  tmp_mat.block<3, 3>(0, 0) = Eigen::Quaternion<double>(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
  tmp_mat.block<3, 1>(0, 3) = Eigen::Matrix<double, 3, 1>(pose.position.x, pose.position.y, pose.position.z);

  {
    lock_guard<mutex> lock(m_mutex_pose);
    m_pose_input = make_pair(msg->header.stamp.toSec(), tmp_mat);
  }
}

void map_representation_timer_func(const ros::TimerEvent& event){
  bool if_time_sync=false;
  pcl::PointCloud<pcl::PointXYZ> pcl_input, pcl_input_map;
  Eigen::Matrix<double, 4, 4> map_t_sensor;
  {
    lock_guard<mutex> lock(m_mutex_pcl);
    {
      lock_guard<mutex> lock(m_mutex_pose);
      if (fabs(m_pose_input.first - m_pcl_input.first) < 0.03) //time synchronized
      {
        if_time_sync=true;
        map_t_sensor = m_pose_input.second * m_body_t_lidar;
        pcl_input = m_pcl_input.second;
      }
    }
  }


  if (if_time_sync)
  {
    pcl::transformPointCloud(pcl_input, pcl_input_map, map_t_sensor);

    octomap::Pointcloud temp_pcl; 
    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator it = pcl_input_map.begin(); it!=pcl_input_map.end(); ++it){
      temp_pcl.push_back(it->x, it->y, it->z);
    } 
    m_octree->insertPointCloud(temp_pcl, octomap::point3d( map_t_sensor(0,3), map_t_sensor(1,3), map_t_sensor(2,3)) );

    pcl::PointCloud<pcl::PointXYZ>::Ptr octo_pcl_pub(new pcl::PointCloud<pcl::PointXYZ>());
    for (octomap::OcTree::iterator it=m_octree->begin(); it!=m_octree->end(); ++it){
      if(m_octree->isNodeOccupied(*it))
      {
        octo_pcl_pub->push_back(pcl::PointXYZ(it.getCoordinate().x(), it.getCoordinate().y(), it.getCoordinate().z()));
      }
    }
    m_pcl_debug_pub.publish(cloud2msg(*octo_pcl_pub));
  }
}

int main(int argc, char **argv){

    signal(SIGINT, signal_handler); // to exit program when ctrl+c


    ros::init(argc, argv, "pcd_generating_node");
    ros::NodeHandle nh("~");

    nh.param("/pcd_voxel_size", m_pcd_voxel_size, 0.3);
    nh.getParam("/lidar_tf", m_lidar_tf);
    if (m_lidar_tf.size()<3)
    {
      m_lidar_tf.push_back(0.1); //z
      m_lidar_tf.push_back(0.0); //y
      m_lidar_tf.push_back(0.0); //x
    }
    m_body_t_lidar.block<3, 1>(0, 3) = Eigen::Matrix<double, 3, 1>(m_lidar_tf[0], m_lidar_tf[1], m_lidar_tf[2]);
    m_octree = make_shared<octomap::OcTree>(m_pcd_voxel_size);
    m_pcl_debug_pub = nh.advertise<sensor_msgs::PointCloud2>("/pcl", 3);
    ros::Subscriber m_pcl_sub = nh.subscribe<sensor_msgs::PointCloud2>("/os_cloud_node/points", 3, &pcl_cb);
    ros::Subscriber m_pose_sub = nh.subscribe<nav_msgs::Odometry>("/mavros/local_position/odom", 3, &pose_cb);
    ros::Subscriber save_sub = nh.subscribe<std_msgs::Empty>("/save_pcd", 3, &save_cb);
    ros::Timer map_timer = nh.createTimer(ros::Duration(1/8.4), &map_representation_timer_func);

    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}