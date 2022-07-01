#ifndef VOXBLOX_TSDF_ESDF_IN_USE_H
#define VOXBLOX_TSDF_ESDF_IN_USE_H

///// common
#include <memory>
#include <queue>
#include <string>
///// Eigen
#include <Eigen/Eigen>
///// pcl
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
///// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
///// voxblox core
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/esdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/utils/color_maps.h>
///// voxblox msg
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/Mesh.h>
#include <voxblox_msgs/Layer.h>
///// voxblox ros
#include <voxblox_ros/ros_params.h>
#include <voxblox_ros/mesh_vis.h>
#include <voxblox_ros/ptcloud_vis.h>


///////////////////////////////////////////////////////////////////////
namespace voxblox {

	constexpr float kDefaultMaxIntensity = 150.0;

	class TsdfEsdfInUse {
	  public:
  	  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  	  TsdfEsdfInUse(const ros::NodeHandle& nh_private, const double &voxel_size);
  	  TsdfEsdfInUse(const ros::NodeHandle& nh_private, const double &voxel_size,
  	             EsdfMap::Config esdf_config,
  	             const EsdfIntegrator::Config& esdf_integrator_config,
  	             TsdfMap::Config config,
  	             const TsdfIntegratorBase::Config& integrator_config,
  	             const MeshIntegratorConfig& mesh_config);
  	  ~TsdfEsdfInUse() {}

  	  template <typename PointType>
  	  void insertPointcloud(const pcl::PointCloud<PointType>& pointcloud, const Eigen::Matrix<float, 4, 4> &map_to_sensor_tf);

  	  void getTsdfSurfacePointsForPublish(pcl::PointCloud<pcl::PointXYZRGB> &pointcloud);
  	  void getTsdfMeshForPublish(voxblox_msgs::Mesh &mesh_msg);
  	  void getPcdNormalFromMeshForPublish(pcl::PointCloud<pcl::PointNormal> &pcd_normals);
      void getEsdfSlice(const double &slice_height, pcl::PointCloud<pcl::PointXYZI> &pointcloud);
  	  size_t getTsdfSize();
  	  size_t getTsdfUsedMemory();
  	  
  	  void updateMesh(); /// Incremental update.
  	  bool generateMesh(); /// Batch update and save mesh file
      void updateEsdf(); //update the ESDF based on latest state of the TSDF map, only the newly updated parts of the TSDF map
  	  void clear(); /// CLEARS THE ENTIRE MAP!

  	  bool clearMapCallback(std_srvs::Empty::Request& request,           // NOLINT
  	                        std_srvs::Empty::Response& response);        // NOLINT
  	  bool generateMeshCallback(std_srvs::Empty::Request& request,       // NOLINT
                              	std_srvs::Empty::Response& response);    // NOLINT

      inline std::shared_ptr<TsdfMap> getTsdfMapPtr() { return tsdf_map_; }
      inline std::shared_ptr<const TsdfMap> getTsdfMapPtr() const { return tsdf_map_; }
      inline std::shared_ptr<EsdfMap> getEsdfMapPtr() { return esdf_map_; }
      inline std::shared_ptr<const EsdfMap> getEsdfMapPtr() const { return esdf_map_; }

	  protected:
      ros::NodeHandle nh_private_;
      // Services.
      ros::ServiceServer clear_map_srv_;
      ros::ServiceServer generate_mesh_srv_;
      /// Delete blocks that are far from the system to help manage memory
      double max_block_distance_from_body_=1e4;
      /// Mesh output settings. Mesh is only written to file if mesh_filename_ is not empty.
      std::string mesh_filename_;
      /// How to color the mesh.
      ColorMode color_mode_;
      /// Colormap to use for intensity pointclouds.
      std::shared_ptr<ColorMap> color_map_;
      // TSDF Maps and integrators.
      std::shared_ptr<TsdfMap> tsdf_map_;
      std::unique_ptr<TsdfIntegratorBase> tsdf_integrator_;
      // Mesh accessories.
      std::shared_ptr<MeshLayer> mesh_layer_;
      std::unique_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;
      // ESDF Maps and integrators.
      std::shared_ptr<EsdfMap> esdf_map_;
      std::unique_ptr<EsdfIntegrator> esdf_integrator_;
      // Pose in minkindr_ros
      Transformation T_G_C;
	};


}  // namespace voxblox








//////////////////////////////////////////////////////////////////////////////
namespace voxblox {
////////////////// basinc funcs
	TsdfEsdfInUse::TsdfEsdfInUse(const ros::NodeHandle& nh_private, const double &voxel_size)
	    					 : TsdfEsdfInUse(nh_private, voxel_size,
	    	    		 	 getEsdfMapConfigFromRosParam(nh_private),
                   getEsdfIntegratorConfigFromRosParam(nh_private),
                   getTsdfMapConfigFromRosParam(nh_private),
                   getTsdfIntegratorConfigFromRosParam(nh_private),
                   getMeshIntegratorConfigFromRosParam(nh_private)) {}

	TsdfEsdfInUse::TsdfEsdfInUse(const ros::NodeHandle& nh_private, const double &voxel_size,
												 EsdfMap::Config esdf_config,
                       	 const EsdfIntegrator::Config& esdf_integrator_config,
	                       TsdfMap::Config config,
	                       const TsdfIntegratorBase::Config& integrator_config,
	                       const MeshIntegratorConfig& mesh_config)
                  	    : nh_private_(nh_private),
                  	      color_map_(new RainbowColorMap())
  {

    // Mesh settings.
    nh_private.param("mesh_filename", mesh_filename_, mesh_filename_);
    std::string color_mode("");
    nh_private.param("color_mode", color_mode, color_mode);
    color_mode_ = getColorModeFromString(color_mode);

    // Color map for intensity pointclouds.
    std::string intensity_colormap("rainbow");
    float intensity_max_value = kDefaultMaxIntensity;
    nh_private.param("intensity_colormap", intensity_colormap, intensity_colormap);
    nh_private.param("intensity_max_value", intensity_max_value, intensity_max_value);

    // Default set in constructor.
    if (intensity_colormap == "rainbow") {
      color_map_.reset(new RainbowColorMap());
    }
    else if (intensity_colormap == "inverse_rainbow") {
      color_map_.reset(new InverseRainbowColorMap());
    }
    else if (intensity_colormap == "grayscale") {
      color_map_.reset(new GrayscaleColorMap());
    }
    else if (intensity_colormap == "inverse_grayscale") {
      color_map_.reset(new InverseGrayscaleColorMap());
    }
    else if (intensity_colormap == "ironbow") {
      color_map_.reset(new IronbowColorMap());
    }
    else {
      ROS_ERROR_STREAM("Invalid color map: " << intensity_colormap);
    }
    color_map_->setMaxValue(intensity_max_value);

	  // Initialize TSDF Map and integrator.
	  config.tsdf_voxel_size = static_cast<float>(voxel_size);
	  tsdf_map_.reset(new TsdfMap(config));

	  std::string method("merged");
	  nh_private_.param("method", method, method);
	  if (method.compare("simple") == 0) {
	    tsdf_integrator_.reset( new SimpleTsdfIntegrator(integrator_config, tsdf_map_->getTsdfLayerPtr()) );
	  }
    else if (method.compare("merged") == 0) {
	    tsdf_integrator_.reset( new MergedTsdfIntegrator(integrator_config, tsdf_map_->getTsdfLayerPtr()) );
	  }
    else if (method.compare("fast") == 0) {
	    tsdf_integrator_.reset( new FastTsdfIntegrator(integrator_config, tsdf_map_->getTsdfLayerPtr()) );
	  }
    else {
	    tsdf_integrator_.reset( new SimpleTsdfIntegrator(integrator_config, tsdf_map_->getTsdfLayerPtr()) );
	  }

	  mesh_layer_.reset( new MeshLayer(tsdf_map_->block_size()) );
	  mesh_integrator_.reset( new MeshIntegrator<TsdfVoxel>(mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()) );

    // Set up map and integrator.
    esdf_config.esdf_voxel_size = static_cast<float>(voxel_size);
    esdf_map_.reset( new EsdfMap(esdf_config) );
    esdf_integrator_.reset( new EsdfIntegrator(esdf_integrator_config,tsdf_map_->getTsdfLayerPtr(),esdf_map_->getEsdfLayerPtr()) );

	  // Advertise services.
	  generate_mesh_srv_ = nh_private_.advertiseService("generate_mesh", &TsdfEsdfInUse::generateMeshCallback, this);
	  clear_map_srv_ = nh_private_.advertiseService("clear_map", &TsdfEsdfInUse::clearMapCallback, this);
	}





////////////////// in use funcs
	template <typename PointType>
	void TsdfEsdfInUse::insertPointcloud(const pcl::PointCloud<PointType>& pointcloud, const Eigen::Matrix<float, 4, 4> &map_to_sensor_tf){

	  Eigen::Matrix<float, 3, 1> position = map_to_sensor_tf.block<3, 1>(0, 3);
    Eigen::Quaternion<float> rotation(map_to_sensor_tf.block<3, 3>(0, 0));
		T_G_C = kindr::minimal::QuatTransformationTemplate<float>(rotation, position);

	  // Convert the PCL pointcloud into our awesome format.
	  Pointcloud points_C;
	  Colors colors;
    convertPointcloud(pointcloud, color_map_, &points_C, &colors);

    // integrate pointcloud
	  CHECK_EQ(points_C.size(), colors.size());
	  tsdf_integrator_->integratePointCloud(T_G_C, points_C, colors);

	  // updating Tsdf and mesh layers
	  tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(T_G_C.getPosition(), max_block_distance_from_body_);
	  mesh_layer_->clearDistantMesh(T_G_C.getPosition(), max_block_distance_from_body_);
	}

  void TsdfEsdfInUse::updateEsdf() {
    if (tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() > 0) {
      const bool clear_updated_flag_esdf = true;
      esdf_integrator_->updateFromTsdfLayer(clear_updated_flag_esdf);
    }
    esdf_map_->getEsdfLayerPtr()->removeDistantBlocks(T_G_C.getPosition(), max_block_distance_from_body_);
  }

	void TsdfEsdfInUse::updateMesh() {
	  constexpr bool only_mesh_updated_blocks = true;
	  constexpr bool clear_updated_flag = true;
	  mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
	}


	void TsdfEsdfInUse::clear() {
    // Esdf clear
    esdf_map_->getEsdfLayerPtr()->removeAllBlocks();
    esdf_integrator_->clear();
    CHECK_EQ(esdf_map_->getEsdfLayerPtr()->getNumberOfAllocatedBlocks(), 0u);

    // Tsdf clear
	  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
	  mesh_layer_->clear();
	}

	void TsdfEsdfInUse::getTsdfSurfacePointsForPublish(pcl::PointCloud<pcl::PointXYZRGB> &pointcloud) {
	  // Create a pointcloud with distance = intensity.
	  const float surface_distance_thresh = tsdf_map_->getTsdfLayer().voxel_size() * 0.75;
	  createSurfacePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(), surface_distance_thresh, &pointcloud);
	}

	void TsdfEsdfInUse::getPcdNormalFromMeshForPublish(pcl::PointCloud<pcl::PointNormal> &pcd_normals){
		Mesh combined_mesh;
		mesh_layer_->getConnectedMesh(&combined_mesh);

		pcd_normals.clear();
		for (size_t i = 0; i < combined_mesh.vertices.size(); ++i)
		{
			Eigen::Matrix<float, 3, 1> pt = combined_mesh.vertices[i];
			Eigen::Matrix<float, 3, 1> normal = combined_mesh.normals[i];
			pcl::PointNormal temp_ptnorm;
			temp_ptnorm.x = pt(0);
			temp_ptnorm.y = pt(1);
			temp_ptnorm.z = pt(2);
			temp_ptnorm.normal[0] = normal(0);
			temp_ptnorm.normal[1] = normal(1);
			temp_ptnorm.normal[2] = normal(2);
			pcd_normals.push_back(temp_ptnorm);
		}
	}

	void TsdfEsdfInUse::getTsdfMeshForPublish(voxblox_msgs::Mesh &mesh_msg){
	  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg);
	}
  void TsdfEsdfInUse::getEsdfSlice(const double &slice_height, pcl::PointCloud<pcl::PointXYZI> &pointcloud){
    constexpr int kZAxisIndex = 2;
    createDistancePointcloudFromEsdfLayerSlice( esdf_map_->getEsdfLayer(), kZAxisIndex, slice_height, &pointcloud );
  }

	size_t TsdfEsdfInUse::getTsdfSize(){
		return tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks();
	}
  size_t TsdfEsdfInUse::getTsdfUsedMemory(){
  	return tsdf_map_->getTsdfLayer().getMemorySize();
  }



////////////////// service funcs
	bool TsdfEsdfInUse::generateMesh() {
	  const bool clear_mesh = true;
	  if (clear_mesh) {
	    constexpr bool only_mesh_updated_blocks = false;
	    constexpr bool clear_updated_flag = true;
	    mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
	  }
    else {
	    constexpr bool only_mesh_updated_blocks = true;
	    constexpr bool clear_updated_flag = true;
	    mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
	  }
	  voxblox_msgs::Mesh mesh_msg_not_used_;
	  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg_not_used_);

	  if (!mesh_filename_.empty()) {
	    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
	    if (success) {
	      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
	    } else {
	      ROS_INFO("Failed to output mesh as PLY: %s", mesh_filename_.c_str());
	    }
	  }

	  return true;
	}

	bool TsdfEsdfInUse::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
	                                      std_srvs::Empty::Response& /*response*/) {  // NOLINT
	  return generateMesh();
	}

	bool TsdfEsdfInUse::clearMapCallback(std_srvs::Empty::Request& /*request*/,
	                                  std_srvs::Empty::Response& /*response*/) {  // NOLINT
	  clear();
	  return true;
	}

	

}  // namespace voxblox


#endif  // VOXBLOX_TSDF_ESDF_IN_USE_H