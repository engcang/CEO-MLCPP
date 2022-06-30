#ifndef CEO_MLCPP_MAIN_H
#define CEO_MLCPP_MAIN_H


#include "utilities.h"
#include "ikd_Tree.h"

///// C++
#include <random>
#include <algorithm>
#include <signal.h>
#include <string>
#include <sstream>
#include <math.h>
#include <chrono>


///// Eigen, Linear Algebra
#include <Eigen/Eigen> //whole Eigen library

///// OpenCV
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

///// ROS
#include <ros/ros.h>
#include <tf_conversions/tf_eigen.h> // to Quaternion_to_euler
#include <tf/LinearMath/Quaternion.h> // to Quaternion_to_euler
#include <tf/LinearMath/Matrix3x3.h> // to Quaternion_to_euler
#include <std_msgs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Path.h>

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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
//normal
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>


using namespace std;
using PointType = pcl::PointXYZ;
using PointVectorIKdTree = KD_TREE<PointType>::PointVector;


//// main class
class ceo_mlcpp_class{
  public:
    ///// basic params
    bool m_pcd_load=false, m_pre_process=false, m_traj_refined_check=false;
    bool m_debug_mode=false;
    string m_infile;
    vector<double> m_cam_intrinsic, m_cam_extrinsic;
    Eigen::Matrix4d m_body_t_cam = Eigen::Matrix4d::Identity();

    ///// MLCPP params
    double m_max_dist = 15.0;
    double m_max_angle = 60.0;
    double m_view_pt_dist = 10.0; //from points
    double m_view_pt_each_dist = 2.0; //between each viewpoints
    double m_view_overlap = 0.1; //overlap bet two viewpoints
    double m_slice_height = 8.0;
    int m_TSP_trial = 100;
    ///// CEO-MLCPP params
    double m_max_velocity = 1.0;
    double m_collision_radius = 1.0;
    ///// MLCPP variables
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> m_normal_estimator;
    pcl::PointXYZ m_pcd_center_point;
    pcl::PointCloud<pcl::PointXYZ> m_cloud_map, m_cloud_center, m_cloud_none_viewed;
    pcl::PointCloud<pcl::PointXYZ> m_cloud_initial_view_point, m_optimized_view_point;
    pcl::PointCloud<pcl::PointNormal> m_cloud_normals;
    geometry_msgs::PoseArray m_normal_pose_array;
    nav_msgs::Path m_all_layer_path, m_all_layer_refined_path;

    ///// ikd-Tree
    shared_ptr<KD_TREE<PointType>> m_ikd_Tree = nullptr;

    ///// ROS
    ros::NodeHandle m_nh;
    ros::Subscriber m_path_calc_sub;
    ros::Publisher m_cloud_map_pub, m_cloud_center_pub, m_cloud_none_viewed_pub;
    ros::Publisher m_initial_view_point_pub, m_optimized_view_point_pub;
    ros::Publisher m_cloud_normal_pub, m_all_layer_path_pub, m_all_layer_path_refined_pub;
    ros::Timer m_visualizing_timer;

    ///// Functions    
    // ROS
    void calc_cb(const std_msgs::Empty::ConstPtr& msg);
    void visualizer_timer_func(const ros::TimerEvent& event);
    // init
    void load_pcd();
    void preprocess_pcd();
    // funcs
    bool check_cam_in(Eigen::VectorXd view_point_xyzpy,pcl::PointXYZ point,pcl::Normal normal);
    void flip_normal(pcl::PointXYZ base,pcl::PointXYZ center,float & nx,float & ny, float & nz);
    void TwoOptSwap(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, int start, int finish);
    double PclArrayCost(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, double &distance);
    double TwoOptTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray);
    // traj refine
    void traj_refinement(const nav_msgs::Path &path_in, nav_msgs::Path &path_out);
    // collision
    bool collision_line(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const double &collision_radius);

    //constructor
    ceo_mlcpp_class(const ros::NodeHandle& n);
    ~ceo_mlcpp_class(){};
};









////////////////// can be separated into .cpp file
///// class constructor
ceo_mlcpp_class::ceo_mlcpp_class(const ros::NodeHandle& n) : m_nh(n){
  // params
  m_nh.param<string>("/infile", m_infile, "resource/1875935000.pcd");
  m_nh.param<bool>("/debug_mode", m_debug_mode, false);
  m_nh.getParam("/cam_intrinsic", m_cam_intrinsic);
  m_nh.getParam("/cam_extrinsic", m_cam_extrinsic);
  m_nh.param("/slice_height", m_slice_height, 8.0);
  m_nh.param("/max_dist", m_max_dist, 15.0);
  m_nh.param("/max_angle", m_max_angle, 60.0);
  m_nh.param("/view_pt_dist", m_view_pt_dist, 10.0);
  m_nh.param("/view_pt_each_dist", m_view_pt_each_dist, 2.0);
  m_nh.param("/view_overlap", m_view_overlap, 0.1);
  m_nh.param("/TSP_trial", m_TSP_trial, 100);
  m_nh.param("/max_velocity", m_max_velocity, 1.0);
  m_nh.param("/collision_radius", m_collision_radius, 10.0);

  m_body_t_cam.block<3, 3>(0, 0) = Eigen::Quaterniond(-0.5, 0.5, -0.5, 0.5).toRotationMatrix(); //TODO
  m_body_t_cam.block<3, 1>(0, 3) = Eigen::Vector3d(m_cam_extrinsic[0], m_cam_extrinsic[1], m_cam_extrinsic[2]);

  //sub
  m_path_calc_sub = m_nh.subscribe<std_msgs::Empty>("/calculate_cpp", 3, &ceo_mlcpp_class::calc_cb, this);

  //pub
  m_cloud_map_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/pcl_map", 3);
  m_cloud_center_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/pcl_center", 3);
  m_cloud_none_viewed_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/none_viewed_pcl", 3);
  m_initial_view_point_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/initial_viewpoints", 3);
  m_optimized_view_point_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/optimized_viewpoints", 3);
  m_cloud_normal_pub = m_nh.advertise<geometry_msgs::PoseArray>("/pcl_normals", 3);
  m_all_layer_path_pub = m_nh.advertise<nav_msgs::Path>("/ceo_mlcpp_path", 3);
  m_all_layer_path_refined_pub = m_nh.advertise<nav_msgs::Path>("/ceo_mlcpp_refined_path", 3);

  //ikdtree
  m_ikd_Tree = make_shared<KD_TREE<PointType>>();

  //timer
  m_visualizing_timer = m_nh.createTimer(ros::Duration(1/5.0), &ceo_mlcpp_class::visualizer_timer_func, this);

  //init
  load_pcd(); //Get Map from pcd
  preprocess_pcd(); //Preprocess pcd: ground removal, normal estimation
}



///// functions
void ceo_mlcpp_class::load_pcd(){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  m_cloud_map.clear();
  m_cloud_center.clear();
  m_cloud_normals.clear();
  m_cloud_none_viewed.clear();
  m_cloud_initial_view_point.clear();
  m_optimized_view_point.clear();

  ROS_INFO("loading %s", m_infile.c_str());
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (m_infile.c_str (), m_cloud_map) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read pcd file \n");
    return;
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;

  ROS_WARN("Map successfully obtained from PCD in %.3f [ms]", duration);

  m_pcd_load = true;
}

void ceo_mlcpp_class::preprocess_pcd(){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  ////// Ground Eleminate
  ROS_INFO("Ground filtering Start!");
  m_pcd_center_point.x = 0; m_pcd_center_point.y = 0; m_pcd_center_point.z = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map_nogr(new pcl::PointCloud<pcl::PointXYZ>);
  for(size_t i = 0; i < m_cloud_map.points.size() ; ++i){
    pcl::PointXYZ point;
    point.x = m_cloud_map.points[i].x;
    point.y = m_cloud_map.points[i].y;
    point.z = m_cloud_map.points[i].z;
    if(point.z > 0.4){
      Eigen::Vector4d vec;
      vec<<point.x, point.y, point.z, 1;
      cloud_map_nogr->points.push_back(point);
      m_pcd_center_point.x += point.x;
      m_pcd_center_point.y += point.y;
      m_pcd_center_point.z += point.z;
    }
  }
  m_pcd_center_point.x = m_pcd_center_point.x / cloud_map_nogr->points.size();
  m_pcd_center_point.y = m_pcd_center_point.y / cloud_map_nogr->points.size();
  m_pcd_center_point.z = m_pcd_center_point.z / cloud_map_nogr->points.size();
  m_cloud_center.push_back(m_pcd_center_point);

  m_cloud_map.clear();
  m_cloud_map = *cloud_map_nogr;
  m_cloud_map.width = m_cloud_map.points.size();
  m_cloud_map.height = 1;
  ROS_INFO("Ground filtering Finished!");



  ////// Normal estimation
  ROS_INFO("Normal Estimation Start!");
  m_cloud_normals.clear();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  *cloud_in = m_cloud_map;
  m_normal_estimator.setInputCloud(cloud_in);
  m_normal_estimator.setSearchMethod (tree);
  // m_normal_estimator.setKSearch (m_pcd_normal_search);
  m_normal_estimator.setRadiusSearch (4.0); //TODO, parameterlize
  m_normal_estimator.compute (*cloud_normals);
  for (size_t i=0; i<cloud_normals->points.size(); ++i)
  {
    flip_normal(cloud_in->points[i], m_pcd_center_point, cloud_normals->points[i].normal[0], cloud_normals->points[i].normal[1], cloud_normals->points[i].normal[2]);
    pcl::PointNormal temp_ptnorm;
    temp_ptnorm.x = cloud_in->points[i].x;
    temp_ptnorm.y = cloud_in->points[i].y;
    temp_ptnorm.z = cloud_in->points[i].z;
    temp_ptnorm.normal[0] = cloud_normals->points[i].normal[0];
    temp_ptnorm.normal[1] = cloud_normals->points[i].normal[1];
    temp_ptnorm.normal[2] = cloud_normals->points[i].normal[2];
    m_cloud_normals.push_back(temp_ptnorm);
  }
  m_cloud_normals.width = m_cloud_normals.points.size();
  m_cloud_normals.height = 1;
  ROS_INFO("Normal Estimation Finish!");
  ROS_INFO("Cloud size : %lu",cloud_in->points.size());
  ROS_INFO("Cloud Normal size : %lu",cloud_normals->points.size());

  m_normal_pose_array = pclnormal_to_posearray(m_cloud_normals);
  m_normal_pose_array.header.frame_id = "map";


  ///// ikd-Tree for collision
  PointVectorIKdTree pcl_input_map_point_vector_ikd;
  for (int i = 0; i < m_cloud_map.size(); ++i)
  {
    pcl_input_map_point_vector_ikd.push_back(m_cloud_map.points[i]);
  }
  m_ikd_Tree->Build(pcl_input_map_point_vector_ikd);
  ROS_INFO("ikd-Tree updated");


  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("PCL preprocessed in %.3f [ms]", duration);

  m_pre_process=true;	
}


void ceo_mlcpp_class::calc_cb(const std_msgs::Empty::ConstPtr& msg){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	if (m_pcd_load && m_pre_process){
    m_cloud_initial_view_point.clear();
    m_cloud_none_viewed.clear();
    m_optimized_view_point.clear();
    m_all_layer_path.header.stamp = ros::Time::now();
    m_all_layer_path.header.frame_id = "map";
    m_all_layer_path.poses.clear();
    m_traj_refined_check=false;
    std::random_device rd_;
    std::mt19937 random_func(rd_());
    
    ////// calculate Coverage Path
    ///Make Initial viewpoint
    bool finish = false;
    int current_layer=1;
    pcl::PointNormal minpt;
    pcl::PointNormal maxpt;
    pcl::getMinMax3D(m_cloud_normals, minpt, maxpt);
    float minpt_z = minpt.z;
    ///PCL Slice with Z axis value (INITIAL)
    pcl::PointCloud<pcl::PointNormal>::Ptr Sliced_target_points_normals_all(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr all_points_normals_temporal(new pcl::PointCloud<pcl::PointNormal>);
    *all_points_normals_temporal = m_cloud_normals;
    pcl::PassThrough<pcl::PointNormal> pass;
    pass.setInputCloud (all_points_normals_temporal);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (minpt_z,minpt_z+m_slice_height);
    pass.filter (*Sliced_target_points_normals_all);

    while(!finish)
    {
      if (minpt_z+m_slice_height >= maxpt.z){ // Check if last layer
        finish = true;
      }
      ///PCL make viewpoints by points and normals
      pcl::PointCloud<pcl::PointNormal>::Ptr viewpoint_ptnorm_temporal(new pcl::PointCloud<pcl::PointNormal>);
      for(int i=0;i<Sliced_target_points_normals_all->points.size();i++)
      {
        pcl::PointNormal temp_ptnorm;
        temp_ptnorm.x = Sliced_target_points_normals_all->points[i].x + Sliced_target_points_normals_all->points[i].normal[0] * m_view_pt_dist;
        temp_ptnorm.y = Sliced_target_points_normals_all->points[i].y + Sliced_target_points_normals_all->points[i].normal[1] * m_view_pt_dist;
        temp_ptnorm.z = Sliced_target_points_normals_all->points[i].z + Sliced_target_points_normals_all->points[i].normal[2] * m_view_pt_dist;
        if(temp_ptnorm.z <= 0 ) continue;
        ///// Path's viewpoint direction should be opposite to normal
        temp_ptnorm.normal[0] = -Sliced_target_points_normals_all->points[i].normal[0];
        temp_ptnorm.normal[1] = -Sliced_target_points_normals_all->points[i].normal[1];
        temp_ptnorm.normal[2] = -Sliced_target_points_normals_all->points[i].normal[2];
        viewpoint_ptnorm_temporal->push_back(temp_ptnorm);
      }
      ///PCL downsample viewpoints with VoxelGrid
      pcl::VoxelGrid<pcl::PointNormal> voxgrid;
      pcl::PointCloud<pcl::PointNormal>::Ptr sliced_view_points (new pcl::PointCloud<pcl::PointNormal>);
      pcl::PointCloud<pcl::PointNormal>::Ptr Sliced_admitted_view_points (new pcl::PointCloud<pcl::PointNormal>);
      voxgrid.setInputCloud(viewpoint_ptnorm_temporal);
      voxgrid.setLeafSize(m_view_pt_each_dist,m_view_pt_each_dist,m_view_pt_each_dist);
      voxgrid.filter(*sliced_view_points);

      ROS_WARN("current layer %d, not viewed, voxelized initial viewpoints: %d", current_layer, sliced_view_points->points.size());
      for (int idx = 0; idx < sliced_view_points->points.size(); ++idx)
      {
        pcl::PointXYZ initial_view_pts;
        initial_view_pts.x = sliced_view_points->points[idx].x;
        initial_view_pts.y = sliced_view_points->points[idx].y;
        initial_view_pts.z = sliced_view_points->points[idx].z;
        m_cloud_initial_view_point.push_back(initial_view_pts);
      }

      pcl::PointCloud<pcl::PointNormal>::Ptr Sliced_target_points_normals_nonviewed (new pcl::PointCloud<pcl::PointNormal>);
      pcl::copyPointCloud(*Sliced_target_points_normals_all,*Sliced_target_points_normals_nonviewed);
      ///PCL downsample viewpoints by view calculation
      int admitted=0;
      int a[sliced_view_points->points.size()];
      for(int i=0;i<sliced_view_points->points.size();i++){
        a[i]=i;
      }
      shuffle(&a[0], &a[sliced_view_points->points.size()], random_func);
      for(int k=0;k<sliced_view_points->points.size();k++)
      {
        int i = a[k];
        vector<int> toerase;
        vector<int> view_comp_map;
        Eigen::VectorXd viewpt(5);
        viewpt << sliced_view_points->points[i].x,sliced_view_points->points[i].y,sliced_view_points->points[i].z,
                  sliced_view_points->points[i].normal[1], sliced_view_points->points[i].normal[2];
                
        for(int j=0;j<Sliced_target_points_normals_nonviewed->points.size();j++)
        {
          pcl::PointXYZ point_toview(Sliced_target_points_normals_nonviewed->points[j].x,Sliced_target_points_normals_nonviewed->points[j].y,
                                     Sliced_target_points_normals_nonviewed->points[j].z);
          pcl::Normal point_normal(Sliced_target_points_normals_nonviewed->points[j].normal[0],
                                   Sliced_target_points_normals_nonviewed->points[j].normal[1],
                                   Sliced_target_points_normals_nonviewed->points[j].normal[2]);
          if(check_cam_in(viewpt,point_toview,point_normal))
          {
            toerase.push_back(j);
          }
        }
        for (size_t j=0;j<Sliced_target_points_normals_all->points.size();j++)
        {
          pcl::PointXYZ point_toview(Sliced_target_points_normals_all->points[j].x,Sliced_target_points_normals_all->points[j].y,
                                     Sliced_target_points_normals_all->points[j].z);
          pcl::Normal point_normal(Sliced_target_points_normals_all->points[j].normal[0],
                                   Sliced_target_points_normals_all->points[j].normal[1],
                                   Sliced_target_points_normals_all->points[j].normal[2]);
          if(check_cam_in(viewpt,point_toview,point_normal))
          {
            view_comp_map.push_back(j);
          }
        }
        if(view_comp_map.size()*m_view_overlap < toerase.size())
        {
          sort(toerase.begin(),toerase.end());
          for(int j=toerase.size()-1;j>-1;j--)
          {
            Sliced_target_points_normals_nonviewed->points.erase(Sliced_target_points_normals_nonviewed->points.begin()+toerase[j]);
          }
          if (m_debug_mode){
            ROS_INFO("%d Point Left", Sliced_target_points_normals_nonviewed->points.size());
            ROS_INFO("Viewpoint %d / %d Admitted ", i, sliced_view_points->points.size());
          }
          admitted++;
          Sliced_admitted_view_points->push_back(sliced_view_points->points[i]);
        }
        //else cout<<"Viewpoint "<<i<<"/"<<sliced_view_points->points.size()<<" Not Admitted"<<endl;
      }
      if (m_debug_mode){
        ROS_INFO("Admitted Viewpoint: %d", admitted);
      }
      
      /// Solve TSP among downsampled viewpoints
      double best_distance = TwoOptTSP(Sliced_admitted_view_points);
      if (best_distance < 0){
        ROS_INFO("collision, doing this layer again");
        continue; // collision, do this slice again
      }
      if (Sliced_admitted_view_points->points.size() < 1){
        ROS_INFO("No admitted points, doing this layer again");
        continue; // collision, do this slice again
      }
      if (current_layer > 1){
        Eigen::Vector3d last_layer_end_point(m_all_layer_path.poses.back().pose.position.x, m_all_layer_path.poses.back().pose.position.y, m_all_layer_path.poses.back().pose.position.z);
        Eigen::Vector3d current_first_point(Sliced_admitted_view_points->points[0].x, Sliced_admitted_view_points->points[0].y, Sliced_admitted_view_points->points[0].z);
        Eigen::Vector3d current_end_point(Sliced_admitted_view_points->points[Sliced_admitted_view_points->points.size()-1].x, Sliced_admitted_view_points->points[Sliced_admitted_view_points->points.size()-1].y, Sliced_admitted_view_points->points[Sliced_admitted_view_points->points.size()-1].z);
        if (!collision_line(last_layer_end_point, current_first_point, m_collision_radius)){
          for(int idx=0; idx<Sliced_admitted_view_points->points.size(); ++idx){
            pcl::PointXYZ optimized_viewpt;
            optimized_viewpt.x = Sliced_admitted_view_points->points[idx].x;
            optimized_viewpt.y = Sliced_admitted_view_points->points[idx].y;
            optimized_viewpt.z = Sliced_admitted_view_points->points[idx].z;
            m_optimized_view_point.push_back(optimized_viewpt);
            m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Sliced_admitted_view_points->points[idx]));
          }
        }
        else if(!collision_line(last_layer_end_point, current_end_point, m_collision_radius)){
          for(int idx=Sliced_admitted_view_points->points.size()-1; idx>=0; --idx){
            pcl::PointXYZ optimized_viewpt;
            optimized_viewpt.x = Sliced_admitted_view_points->points[idx].x;
            optimized_viewpt.y = Sliced_admitted_view_points->points[idx].y;
            optimized_viewpt.z = Sliced_admitted_view_points->points[idx].z;
            m_optimized_view_point.push_back(optimized_viewpt);
            m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Sliced_admitted_view_points->points[idx]));
          }
        }
        else{
          ROS_INFO("collision while connecting layers, doing this layer again");
          continue;
        }
      }
      else{ // first layer
        for(int idx=0; idx<Sliced_admitted_view_points->points.size(); ++idx)
        {
          pcl::PointXYZ optimized_viewpt;
          optimized_viewpt.x = Sliced_admitted_view_points->points[idx].x;
          optimized_viewpt.y = Sliced_admitted_view_points->points[idx].y;
          optimized_viewpt.z = Sliced_admitted_view_points->points[idx].z;
          m_optimized_view_point.push_back(optimized_viewpt);
          m_all_layer_path.poses.push_back(single_pclnormal_to_posestamped(Sliced_admitted_view_points->points[idx]));
        }
      }
      ROS_WARN("current layer %d, still none-viewed points: %d among %d in slice", current_layer, Sliced_target_points_normals_nonviewed->points.size(), Sliced_target_points_normals_all->points.size());
      for (int idx = 0; idx < Sliced_target_points_normals_nonviewed->points.size(); ++idx)
      {
        pcl::PointXYZ none_view_pcl;
        none_view_pcl.x = Sliced_target_points_normals_nonviewed->points[idx].x;
        none_view_pcl.y = Sliced_target_points_normals_nonviewed->points[idx].y;
        none_view_pcl.z = Sliced_target_points_normals_nonviewed->points[idx].z;
        m_cloud_none_viewed.push_back(none_view_pcl);
      }
      ROS_WARN("current layer %d, TSP %d points, leng: %.2f m", current_layer, Sliced_admitted_view_points->points.size(), best_distance);
      
      ///PCL Slice with Z axis value (untill maxpt.z)
      minpt_z += m_slice_height;
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (minpt_z,minpt_z+m_slice_height);
      pass.filter (*Sliced_target_points_normals_all);
      current_layer++;
    } //while end

    //// traj refinement
    traj_refinement(m_all_layer_path, m_all_layer_refined_path);
    m_traj_refined_check=true;

	} //if end
	else{
		ROS_WARN("One of cam info / PCD file loading / PCD pre-process has not been done yet");
	}

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
  ROS_WARN("CEO-MLCPP calculation: %.3f [ms]", duration);
}

void ceo_mlcpp_class::visualizer_timer_func(const ros::TimerEvent& event){
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	if (m_pcd_load && m_pre_process){
    m_cloud_map_pub.publish(cloud2msg(m_cloud_map));
    m_cloud_center_pub.publish(cloud2msg(m_cloud_center));
    m_cloud_normal_pub.publish(m_normal_pose_array);
    m_initial_view_point_pub.publish(cloud2msg(m_cloud_initial_view_point));
    m_cloud_none_viewed_pub.publish(cloud2msg(m_cloud_none_viewed));
    m_optimized_view_point_pub.publish(cloud2msg(m_optimized_view_point));
    m_all_layer_path_pub.publish(m_all_layer_path);
    if (m_traj_refined_check)
      m_all_layer_path_refined_pub.publish(m_all_layer_refined_path);
	}

  if(m_debug_mode){
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e3;
    ROS_WARN("visualizing in %.2f [ms]", duration);
  }
}






///////// methods
//TODO: flip_normal not from center but from real view point, where recorded PCL for none-convex targets
void ceo_mlcpp_class::flip_normal(pcl::PointXYZ base, pcl::PointXYZ center, float & nx, float & ny, float & nz)
{
  float xdif = base.x - center.x;
  float ydif = base.y - center.y;
  if(xdif * nx + ydif * ny <0)
  {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
}

bool ceo_mlcpp_class::collision_line(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const double &collision_radius){
  Eigen::Vector3d direction = (p2 - p1).normalized();
  double length = (p2 - p1).norm();

  int check_step = int(3.0 * length / collision_radius);
  // cout << p1.transpose() << " goal: " << p2.transpose() << " len: " << length << " step: " << check_step << endl;

  for (int i = 1; i <= check_step; ++i)
  {
    Eigen::Vector3d search_vec = p1 + direction * length * ((double)i / (double)check_step);
    PointType search_pt{search_vec(0), search_vec(1), search_vec(2)};
    PointVectorIKdTree Storage;
    m_ikd_Tree->Radius_Search(search_pt, collision_radius, Storage);
    if (Storage.size()>0){
      return true;
    }
  }
  return false;
}

bool ceo_mlcpp_class::check_cam_in(Eigen::VectorXd view_point_xyzpy,pcl::PointXYZ point,pcl::Normal normal)
{
  /// if too far
  float dist = sqrt(pow((view_point_xyzpy(0)-point.x),2)+pow((view_point_xyzpy(1)-point.y),2)+pow((view_point_xyzpy(2)-point.z),2));
  if (dist > m_max_dist){
    return false;
  }

  Eigen::Vector3d normal_pt(normal.normal_x,normal.normal_y,normal.normal_z);
  Eigen::Vector3d Normal_view_pt((view_point_xyzpy(0)-point.x), (view_point_xyzpy(1)-point.y), (view_point_xyzpy(2)-point.z));
  double inner_product = Normal_view_pt.dot(normal_pt)/dist;
  double angle = acos(inner_product)/M_PI*180.0;
  /// if angle
  if (fabs(angle)>m_max_angle){
    return false;
  }

  double viewpt_cam_pitch = asin(-view_point_xyzpy(4))/M_PI*180.0;
  double viewpt_cam_yaw = asin(view_point_xyzpy(3)/cos(-view_point_xyzpy(4)))/M_PI*180.0;

  Eigen::Matrix4d world_t_body = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d world_t_body_rot;
  tf::Quaternion view_point_quat;
  view_point_quat.setRPY(0.0, viewpt_cam_pitch, viewpt_cam_yaw);
  tf::Matrix3x3 view_point_rotation_mat(view_point_quat);
  tf::matrixTFToEigen(view_point_rotation_mat, world_t_body_rot);

  world_t_body.block<3, 1>(0, 3) = view_point_xyzpy.block<3, 1>(0, 0);
  world_t_body.block<3, 3>(0, 0) = world_t_body_rot;

  ///// point in cam frame
  Eigen::Vector4d world_frame_pt(point.x, point.y, point.z, 1.0);
  Eigen::Vector4d world_frame_pt_offset = world_t_body.inverse() * world_frame_pt;
  Eigen::Vector4d cam_frame_pt = m_body_t_cam * world_frame_pt_offset;

  ///// point 3d to 2d
  cv::Point2d uv;
  uv.x = m_cam_intrinsic[2]*cam_frame_pt(0)/cam_frame_pt(2) + m_cam_intrinsic[4];
  uv.y = m_cam_intrinsic[3]*cam_frame_pt(1)/cam_frame_pt(2) + m_cam_intrinsic[5];
  uv.x = floor(fabs(uv.x)) * ((uv.x > 0) - (uv.x < 0));
  uv.y = floor(fabs(uv.y)) * ((uv.y > 0) - (uv.y < 0));
  /// if not in FoV
  if (uv.x<0 || uv.x>m_cam_intrinsic[0] || uv.y<0 || uv.y>m_cam_intrinsic[1]){
    return false;
  }
  return true;
}


//TWO OPT ALGORITHM
void ceo_mlcpp_class::TwoOptSwap(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray,int start,int finish)
{
  int size = pclarray->points.size();
  pcl::PointCloud<pcl::PointNormal> temp_Array;
  for(int i=0;i<=start-1;i++) temp_Array.push_back(pclarray->points[i]);
  for(int i=finish;i>=start;i--) temp_Array.push_back(pclarray->points[i]);
  for(int i=finish+1;i<=size-1;i++) temp_Array.push_back(pclarray->points[i]);
  pcl::copyPointCloud(temp_Array,*pclarray);
}

double ceo_mlcpp_class::PclArrayCost(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray, double &distance)
{
  double cost = 0, dist = 0;
  for(int i=1;i<pclarray->points.size();i++)
  {
    double x_dist = pclarray->points[i].x -  pclarray->points[i-1].x;
    double y_dist = pclarray->points[i].y -  pclarray->points[i-1].y;
    double z_dist = pclarray->points[i].z -  pclarray->points[i-1].z;
    dist += sqrt( x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
    cost += dist;

    if (collision_line(Eigen::Vector3d(pclarray->points[i-1].x, pclarray->points[i-1].y, pclarray->points[i-1].z), Eigen::Vector3d(pclarray->points[i].x, pclarray->points[i].y, pclarray->points[i].z), m_collision_radius)){
      cost += 99999.9;
    }
  }
  distance = dist;
  return cost;
}

double ceo_mlcpp_class::TwoOptTSP(pcl::PointCloud<pcl::PointNormal>::Ptr pclarray)
{
  pcl::PointCloud<pcl::PointNormal> temp_Array;
  pcl::copyPointCloud(*pclarray,temp_Array);
  int size = pclarray->points.size();
  int improve = 0;
  double best_distance, best_cost;
  best_cost = PclArrayCost(pclarray, best_distance);
  if (m_debug_mode){
    ROS_INFO("Initial cost: %.2f", best_cost);
  }
  while (improve<m_TSP_trial)
  {
    for ( int i = 1; i <size - 2; i++ )
    {
      for ( int k = i + 1; k < size-2; k++)
      {
        TwoOptSwap( pclarray, i,k );
        double new_distance, new_cost;
        new_cost = PclArrayCost(pclarray, new_distance);
        if ( new_cost < best_cost )
        {
          improve = 0;
          pcl::copyPointCloud(*pclarray,temp_Array);
          best_cost = new_cost;
          best_distance = new_distance;
        }
      }
    }
    improve ++;
  }
  pcl::copyPointCloud(temp_Array,*pclarray);
  if (m_debug_mode){
    ROS_INFO("Final distance: %.2f", best_distance);
    ROS_INFO("TwoOptTSP Finished");
    ROS_WARN("########################### Final COST: %.2f", best_cost);
  }
  if (best_cost >= 99999){
    best_distance = -1.0;
  }
  return best_distance;
}


void ceo_mlcpp_class::traj_refinement(const nav_msgs::Path &path_in, nav_msgs::Path &path_out){
  path_out.header.stamp = ros::Time::now();
  path_out.header.frame_id = "map";
  path_out.poses.clear();

  Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(8,8);
  for (int i = 0; i < path_in.poses.size()-1; ++i)
  {
    geometry_msgs::Point curr, next;
    Eigen::VectorXd x_coeff(8), y_coeff(8), z_coeff(8), b_x_(8), b_y_(8), b_z_(8);

    curr = path_in.poses[i].pose.position;
    next = path_in.poses[i+1].pose.position;
    tf::Quaternion curr_q(path_in.poses[i].pose.orientation.x, path_in.poses[i].pose.orientation.y, path_in.poses[i].pose.orientation.z, path_in.poses[i].pose.orientation.w);
    tf::Quaternion next_q(path_in.poses[i+1].pose.orientation.x, path_in.poses[i+1].pose.orientation.y, path_in.poses[i+1].pose.orientation.z, path_in.poses[i+1].pose.orientation.w);
    tf::Matrix3x3 curr_m(curr_q);
    tf::Matrix3x3 next_m(next_q);
    double _a, _b, curr_yaw, next_yaw, v_yaw;
    curr_m.getRPY(_a, _b, curr_yaw);
    next_m.getRPY(_a, _b, next_yaw);
    v_yaw = atan2(next.y-curr.y, next.x-curr.x);

    double T_ = sqrt(pow(next.x-curr.x, 2) + pow(next.y-curr.y, 2) + pow(next.z-curr.z, 2)) / m_max_velocity;

    A_ << 0, 0, 0, 0, 0, 0, 0, 1,
    pow(T_,7), pow(T_,6), pow(T_,5), pow(T_,4), pow(T_,3), pow(T_,2), T_, 1,
    0, 0, 0, 0, 0, 0, 1, 0,
    7*pow(T_,6), 6*pow(T_,5), 5*pow(T_,4), 4*pow(T_,3), 3*pow(T_,2), 2*T_, 1, 0,
    0, 0, 0, 0, 0, 2, 0, 0,
    42*pow(T_,5), 30*pow(T_,4), 20*pow(T_,3), 12*pow(T_,2), 6*T_, 2, 0, 0,
    0, 0, 0, 0, 6, 0, 0, 0,
    210*pow(T_,4), 120*pow(T_,3), 60*pow(T_,2), 24*T_, 6, 0, 0, 0;    

    if (i < path_in.poses.size()-2){
      geometry_msgs::Point next_next = path_in.poses[i+2].pose.position;
      double v_yaw_next = atan2(next_next.y-next.y, next_next.x-next.x);
      b_x_ << curr.x, next.x, m_max_velocity*cos(v_yaw), m_max_velocity*cos(v_yaw_next), 0, 0, 0, 0;
      b_y_ << curr.y, next.y, m_max_velocity*sin(v_yaw), m_max_velocity*sin(v_yaw_next), 0, 0, 0, 0;
    }
    else{      
      b_x_ << curr.x, next.x, m_max_velocity*cos(v_yaw), 0, 0, 0, 0, 0;
      b_y_ << curr.y, next.y, m_max_velocity*sin(v_yaw), 0, 0, 0, 0, 0;
    }
    b_z_ << curr.z, next.z, 0, 0, 0, 0, 0, 0;
    x_coeff = A_.lu().solve(b_x_);  y_coeff = A_.lu().solve(b_y_);  z_coeff = A_.lu().solve(b_z_);

    for (double t = T_*0.125; t <= T_; t+=T_*0.125)
    {
      geometry_msgs::PoseStamped p;
      p.pose.position.x = x_coeff[0]*pow(t,7) + x_coeff[1]*pow(t,6) + x_coeff[2]*pow(t,5) + x_coeff[3]*pow(t,4) + x_coeff[4]*pow(t,3) + x_coeff[5]*pow(t,2) + x_coeff[6]*t + x_coeff[7];
      p.pose.position.y = y_coeff[0]*pow(t,7) + y_coeff[1]*pow(t,6) + y_coeff[2]*pow(t,5) + y_coeff[3]*pow(t,4) + y_coeff[4]*pow(t,3) + y_coeff[5]*pow(t,2) + y_coeff[6]*t + y_coeff[7];
      p.pose.position.z = z_coeff[0]*pow(t,7) + z_coeff[1]*pow(t,6) + z_coeff[2]*pow(t,5) + z_coeff[3]*pow(t,4) + z_coeff[4]*pow(t,3) + z_coeff[5]*pow(t,2) + z_coeff[6]*t + z_coeff[7];

      double g_yaw = curr_yaw + (next_yaw-curr_yaw) * t/T_;
      tf::Quaternion qqq;
      qqq.setRPY(0,0, g_yaw); //roll=pitch=0 !!!

      p.pose.orientation.x = qqq.getX();
      p.pose.orientation.y = qqq.getY();
      p.pose.orientation.z = qqq.getZ();
      p.pose.orientation.w = qqq.getW();
      path_out.poses.push_back(p);
    }
  }
}



#endif