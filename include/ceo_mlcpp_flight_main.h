#ifndef CEO_MLCPP_FLIGHT_H
#define CEO_MLCPP_FLIGHT_H

#include "utilities.h"
#include "voxblox_in_use.h"

///// common headers
#include <time.h>
#include <math.h>
#include <cmath>
#include <chrono> 
#include <vector>
#include <mutex>
#include <string>
#include <utility> // pair, make_pair

///// Eigen
#include <Eigen/Eigen> // whole Eigen library: Sparse(Linearalgebra) + Dense(Core+Geometry+LU+Cholesky+SVD+QR+Eigenvalues)

///// ROS
#include <ros/ros.h>
#include <ros/package.h> // get path
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf/LinearMath/Matrix3x3.h> // to Quaternion_to_euler
#include <voxblox_msgs/Mesh.h> // to Quaternion_to_euler
// callback two topics at once
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
// MAVROS
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h> //offboarding
#include <mavros_msgs/CommandBool.h> //arming

///// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

///// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
// voxel
#include <pcl/filters/voxel_grid.h>


using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace voxblox;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> rgbd_sync_policy;


///////////////////////////////////////////////////////////////////////////////////
class ceo_mlcpp_flight_class{
	public:

		///// shared data - sensor, pose
		mutex m_mutex_pose;
		Matrix4d m_map_t_body = Matrix4d::Identity();
    Matrix4d m_body_t_cam = Matrix4d::Identity();
    pair<double, Matrix4d> m_pose_input;
    vector<double> m_cam_extrinsic, m_cam_intrinsic;
    double m_scale_factor=1.0;

		///// Drone control
    mavros_msgs::State m_current_state;
    bool m_pose_check=false, m_ctrl_init=false, m_path_check=false;
    ros::Time m_ctrl_start_t;
    nav_msgs::Path m_path_in;
    int m_path_index=0;

    ///// voxblox
    shared_ptr<TsdfEsdfInUse> m_tsdfesdf_voxblox = nullptr;
    double m_voxel_resolution=0.3;
    int m_downsampling_counter=0;
    pcl::VoxelGrid<pcl::PointXYZRGB> m_voxelgrid;
    ///// image_out
    string m_path;
    cv::Mat m_current_rgb_image;

    ///// ros and tf
    ros::NodeHandle nh;
    ros::Subscriber m_state_sub, m_pose_sub, m_path_sub;
    ros::Publisher m_position_controller_pub, m_voxblox_mesh_pub;
    ros::ServiceClient m_arming_client, m_set_mode_client;
    ros::Timer m_controller_timer;

    ///// functions
    void state_cb(const mavros_msgs::State::ConstPtr& msg);
    void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void path_cb(const nav_msgs::Path::ConstPtr& msg);
    void rgb_depth_cb(const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::Image::ConstPtr& rgb_msg);
    void controller_timer_func(const ros::TimerEvent& event);
    void dummy_controller(double forward_x=0.0, double taking_off_z=0.0);


    ceo_mlcpp_flight_class(ros::NodeHandle& n) : nh(n){
      nh.getParam("/cam_extrinsic", m_cam_extrinsic);
      nh.getParam("/cam_intrinsic", m_cam_intrinsic);
      nh.param("/voxel_resolution", m_voxel_resolution, 0.3);

      // camera TF
      m_body_t_cam.block<3, 3>(0, 0) = Quaterniond(-0.5, 0.5, -0.5, 0.5).toRotationMatrix();
      m_body_t_cam.block<3, 1>(0, 3) = Vector3d(m_cam_extrinsic[0], m_cam_extrinsic[1], m_cam_extrinsic[2]);
      m_path = ros::package::getPath("ceo_mlcpp");

      // publishers
      m_position_controller_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 5);
			m_voxblox_mesh_pub = nh.advertise<voxblox_msgs::Mesh>("/reconstructed_mesh", 5);
      // for mavros
      m_arming_client = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
      m_set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
      // subscribers
      m_state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 10, &ceo_mlcpp_flight_class::state_cb, this);
      m_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 3, &ceo_mlcpp_flight_class::pose_cb, this);
      m_path_sub = nh.subscribe<nav_msgs::Path>("/ceo_mlcpp_refined_path", 3, &ceo_mlcpp_flight_class::path_cb, this);
		  static message_filters::Subscriber<sensor_msgs::Image> depth_sub;
		  static message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
		  depth_sub.subscribe(nh, "/d435i/depth/image_raw", 5);
		  rgb_sub.subscribe(nh, "/d435i/depth/rgb_image_raw", 5);

		  static message_filters::Synchronizer<rgbd_sync_policy> depth_rgb_sync_sub(rgbd_sync_policy(5), depth_sub, rgb_sub);
		  depth_rgb_sync_sub.registerCallback(&ceo_mlcpp_flight_class::rgb_depth_cb,this);

      m_ctrl_start_t = ros::Time::now();
      m_tsdfesdf_voxblox = make_shared<TsdfEsdfInUse>(nh, m_voxel_resolution);
      m_voxelgrid.setLeafSize(m_voxel_resolution/2.0, m_voxel_resolution/2.0, m_voxel_resolution/2.0);
      ROS_WARN("Main class heritated, starting node...");

      // timers
      m_controller_timer = nh.createTimer(ros::Duration(1/20.0), &ceo_mlcpp_flight_class::controller_timer_func, this); // every 1/20 second.
    }

};







//////////////// can be seperated into .cpp files

//////////////////////// callbacks
void ceo_mlcpp_flight_class::state_cb(const mavros_msgs::State::ConstPtr& msg){
  m_current_state=*msg;
}

void ceo_mlcpp_flight_class::path_cb(const nav_msgs::Path::ConstPtr& msg){
	if (!m_path_check){
		m_path_in = *msg;
		m_path_check=true;		
	}
}

void ceo_mlcpp_flight_class::pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
  geometry_msgs::Pose pose = msg->pose;
  Matrix4d tmp_mat = Matrix4d::Identity();

  tmp_mat.block<3, 3>(0, 0) = Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
  tmp_mat.block<3, 1>(0, 3) = Vector3d(pose.position.x, pose.position.y, pose.position.z);

  m_map_t_body = tmp_mat;

  {
    lock_guard<mutex> lock(m_mutex_pose);
    m_pose_input = make_pair(msg->header.stamp.toSec(), tmp_mat);
  }

  m_pose_check=true;
}

void ceo_mlcpp_flight_class::rgb_depth_cb(const sensor_msgs::Image::ConstPtr& depth_msg, const sensor_msgs::Image::ConstPtr& rgb_msg){
	if (m_pose_check && m_path_check && m_ctrl_init){
		m_downsampling_counter++;

		Matrix4d map_t_sensor;
		bool if_time_sync=false;
  	cv::Mat depth_img, rgb_img;
	  try {
	    if (depth_msg->encoding=="32FC1"){
	      cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(*depth_msg, "32FC1"); // == sensor_msgs::image_encodings::TYPE_32FC1
	      depth_img = depth_ptr->image;
	      m_scale_factor=1.0;
	    }
	    else if (depth_msg->encoding=="16UC1"){ // uint16_t (stdint.h) or ushort or unsigned_short
	      cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(*depth_msg, "16UC1"); // == sensor_msgs::image_encodings::TYPE_16UC1
	      depth_img = depth_ptr->image;
	      m_scale_factor=1000.0;
	    }

	    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(*rgb_msg, sensor_msgs::image_encodings::BGR8);
    	rgb_img = img_ptr->image;

	    {
	      lock_guard<mutex> lock(m_mutex_pose);
	      if (fabs(m_pose_input.first - depth_msg->header.stamp.toSec()) < 0.03){
	      	map_t_sensor = m_pose_input.second * m_body_t_cam;
	      	if_time_sync=true;
	      }
	    }
	  }
	  catch (cv_bridge::Exception& e) {
	    ROS_ERROR("Error to cvt depth img");
	    return;
	  }
	  if (!rgb_img.empty())
	  	m_current_rgb_image = rgb_img;

	  ////// doin somethin
	  if (m_downsampling_counter%5==0 && if_time_sync){
	  	pcl::PointCloud<pcl::PointXYZRGB> cam_cvt_pcl;
	  	depth_img_to_pcl(depth_img, rgb_img, m_scale_factor, m_cam_intrinsic, cam_cvt_pcl);
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr before_voxelize(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr after_voxelize(new pcl::PointCloud<pcl::PointXYZRGB>());
      *before_voxelize = cam_cvt_pcl;
      m_voxelgrid.setInputCloud(before_voxelize);
      m_voxelgrid.filter(*after_voxelize);
      cam_cvt_pcl = *after_voxelize;

	  	m_tsdfesdf_voxblox->insertPointcloud(cam_cvt_pcl, map_t_sensor.cast<float>());
	  	m_tsdfesdf_voxblox->updateMesh();
	  	voxblox_msgs::Mesh mesh_msg;
      m_tsdfesdf_voxblox->getTsdfMeshForPublish(mesh_msg);
      mesh_msg.header.frame_id = "map";
      m_voxblox_mesh_pub.publish(mesh_msg);
	  }

	}
}

void ceo_mlcpp_flight_class::controller_timer_func(const ros::TimerEvent& event){
	if (m_pose_check && m_path_check){
	  if (!m_ctrl_init){ // intially taking off
	    if(!m_current_state.armed){
	      mavros_msgs::CommandBool arming_command;
	      arming_command.request.value = true;
	      m_arming_client.call(arming_command);
	      ROS_WARN("Arming...");
	      m_ctrl_start_t = ros::Time::now();
	    }
	    else if(m_current_state.mode != "OFFBOARD"){
	      dummy_controller();
	      mavros_msgs::SetMode offboarding_command;
	      offboarding_command.request.custom_mode = "OFFBOARD";
	      m_set_mode_client.call(offboarding_command);
	      ROS_WARN("Offboarding...");
	      m_ctrl_start_t = ros::Time::now();
	    }
	    else{
	      dummy_controller(0.0, 2.5);
	      if (ros::Time::now() - m_ctrl_start_t >= ros::Duration(8.0)){
	        m_ctrl_init=true;
	      }
	      return;
	    }
	  }
	  

		if (euclidean_dist(m_path_in.poses[m_path_index].pose, m_map_t_body.block<3, 1>(0, 3)) < 0.4){
			if ((m_path_index+1)%8==0 && !m_current_rgb_image.empty()){
				cv::imwrite(m_path+"/images/"+to_string(ros::Time::now().toSec())+".jpg", m_current_rgb_image);
			}
			if (m_path_index < m_path_in.poses.size()-1)
				m_path_index++;
		}
	  m_position_controller_pub.publish(m_path_in.poses[m_path_index]);


	}
}


void ceo_mlcpp_flight_class::dummy_controller(double forward_x, double taking_off_z){
  tf::Quaternion qqq;
  geometry_msgs::PoseStamped goal_pose;
  qqq.setRPY(0,0,0); //roll=pitch=0 !!!
  qqq.normalize();
  goal_pose.pose.position.x = m_map_t_body(0,3) + forward_x;
  goal_pose.pose.position.y = m_map_t_body(1,3);
  goal_pose.pose.position.z = taking_off_z;
  goal_pose.pose.orientation.x = qqq.getX();
  goal_pose.pose.orientation.y = qqq.getY();
  goal_pose.pose.orientation.z = qqq.getZ();
  goal_pose.pose.orientation.w = qqq.getW();
  m_position_controller_pub.publish(goal_pose);
}



#endif