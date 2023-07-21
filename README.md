# Note:
+ 2023-07-22: The code is not completed yet. But I have one in my local. I will finish upload within 3 days.

<br>

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
+ 