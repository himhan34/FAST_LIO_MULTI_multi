# FAST-LIO-MULTI
+ This repository is an [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)'s extended version of multi-LiDAR
+ Optionally, user can choose one of bundle update method vs asynchronous update method

## Related video:

<br>

## Dependencies
+ `ROS`, `Ubuntu`, `PCL` >= 1.8, `Eigen` >= 3.3.4
+ [`livox_ros_driver`](https://github.com/Livox-SDK/livox_ros_driver)
```shell
cd ~/your_workspace/src
git clone https://github.com/Livox-SDK/livox_ros_driver
cd ..
catkin build -DCMAKE_BUILD_TYPE=Release
```

## How to build and run
+ Get the code, and then build
```shell
cd ~/your_workspace/src
git clone https://github.com/engcang/FAST_LIO_MULTI

cd ..
catkin build -DCMAKE_BUILD_TYPE=Release
. devel/setup.bash
```
+ Then run
```shell
roslaunch fast_lio_multi run.launch update_method:=bundle
roslaunch fast_lio_multi run.launch update_method:=async
```

<br>

## Update methods: bundle vs asynchronous
+ Bundle update: merge multi LiDAR scans into one pointcloud, and then update
	+ Prevent no scan data input in extreme situation, e.g., high altitude flight of drones
	+ Longer update interval (which may cause drift during aggresive and fast movement)
	+ **NOTE: current code implementation will properly work for LiDARs with same scan rates (e.g., same 10Hz)**
+ Asynchronous update: update the filter whenever LiDAR scan inputs
	+ Shorter update interval
	+ Depending on the sensor configuration, none-scanned data update may occur

<p align="center">
  <img src="imgs/bundle_method.png" width="600"/>
  <img src="imgs/async.png" width="600"/>
  <br>
  <em>Update methods - (upper): Bundle (bottm): Asynchronous</em>
</p>

+ By utilizing the forward and backward propagation structure of FAST-LIO2, each update method is implemented as:
<p align="center">
  <img src="imgs/updates.png" width="600"/>
  <br>
  <em>Update methods - (left): Bundle (right): Asynchronous</em>
</p>

<br>

## Results of each method
+ For two sensor configurations,
	+ Config1: Livox-MID360 x 2EA (each is tilted +143, -143 degree)
	+ Config2: Livox-MID360 x 1EA (0 degree tilted), Livox-AVIA x 1EA (90 degree tilted)

<p align="center">
  <img src="imgs/config1.png" width="300"/>
  <img src="imgs/confg2.png" width="300"/>
  <br>
  <em>Sensor config - (left): config1 (right): config2</em>
</p>

+ 