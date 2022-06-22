# CEO-MLCPP: Control Efficient and Obstacle-aware Multi-Layer Coverage Path Planner for 3D reconstruction with UAVs
+ The purpose of the algorithm is to inspect a known building with an UAV for 3D reconstruction. (Offline method)
+ International Conference on Robot Intelligence Technology and Applications (RiTA), 2022

<br>

#### Original MLCPP
+ TSP, not considering collision nor smoothness of paths
<p align="center">
  <img src="resource/fixed_mlcpp.png" height="450"/>
  <img src="resource/fixed_mlcpp2.png" height="450"/>
  <img src="resource/mlcpp_collision.png" height="350"/>
  <br>
  <em>Original MLCPP. (left): Target and path. (center): full visualization. (right): collisions of path</em>
</p>


<br>

## How to install
+ Install requirements
```bash
$ sudo apt install libgoogle-glog-dev libeigen3-dev
```
+ Install `gcc-9` and `g++-9`
```bash
$ sudo apt-get install libgoogle-glog-dev
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt update
$ sudo apt install gcc-9 g++-9
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

+ Build this Repo
```bash
cd ~/<your_workspace>/src
git clone https://github.com/engcang/CEO-MLCPP.git

cd CEO-MLCPP/include/ivox3d
tar -xvf tbb2018_20170726oss_lin.tgz

cd ~/<your_workspace>
catkin build -DCMAKE_BUILD_TYPE=Release
```

<br>

## How to run
+ Edit some parameters in `main.launch` file
```bash
roslaunch ceo_mlcpp main.launch
rostopic pub /calculate_cpp std_msgs/Empty
```
+ Warning message `Invalid argument passed to canTransform argument source_frame in tf2 frame_ids cannot be empty`
  + Just ignore it. It will disappear after `rostopic pub /calculate_cpp std_msgs/Empty`

<br>

## TODO
+ traj refinement
+ collision in TSP (iVox)
+ Gazebo - Big Ben
+ Auto flight
+ voxblox -> real-time mesh
+ Rviz into two (one for path, one for voxblox)
+ flip-normal fix
