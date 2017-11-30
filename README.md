# ros_opencl_caffe
## Introduction
OpenCL Caffe([clCaffe](https://github.com/01org/caffe/tree/inference-optimize)) is an OpenCL backend for Caffe from Intel&reg;. With inference optimized by Intel&reg; OpenCL, clCaffe can be used in most scene in high performance, like objects inference.

This project is a ROS wrapper for OpenCL Caffe, providing following features:
* A ROS service for objects inference in a ROS image message
* A ROS publisher for objects inference in a ROS image message from a RGB camera
* Demo applications to show the capablilities of ROS service and publisher

## Prerequisite
* An x86_64 computer with Ubuntu 16.04
* ROS Kinetic
* RGB camera, e.g. Intel&reg; RealSense&trade;, Microsoft&reg; Kinect&trade; or standard USB camera

## Environment Setup
* Install ROS Kinetic Desktop-Full ([guide](http://wiki.ros.org/kinetic/Installation/Ubuntu))
* Create a catkin workspace ([guide](http://wiki.ros.org/catkin/Tutorials/create_a_workspace))
* Install clCaffe ([guide](https://github.com/01org/caffe/wiki/clCaffe))
* Create a symbol link in `/opt/clCaffe`
```Shell
  sudo ln -s <clCaffe-path> /opt/clCaffe
```
* Add clCaffe libraries to LD_LIBRARY_PATH.
```Shell
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/clCaffe/build/lib
```
* Install ROS package for different cameras as needed. e.g.
  1. Standard USB camera
  ```Shell
    sudo apt-get install ros-kinetic-usb-cam
  ```
  2. Intel&reg; RealSense&trade; camera
  - Install Intel&reg; RealSense&trade; SDK 2.0 ([guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)). Refer [here](https://github.com/IntelRealSense/librealsense) for more details about Intel&reg; RealSense&trade; SDK 2.0.
  - Install Intel&reg; RealSense&trade; ROS ([guide](https://github.com/intel-ros/realsense))
  ```Shell
    cd ~/catkin_ws/src
    git clone https://github.com/intel-ros/realsense.git
    cd  realsense
    git checkout 2.0.0
    cd ~/catkin_ws
    catkin_make
  ```
  3. Microsoft&reg; Kinect&trade; camera
  ```Shell
    sudo apt-get install ros-kinetic-openni-launch
  ```

## Building and Installation
```Shell
  cd ~/catkin_ws/src
  git clone https://github.com/intel/object_msgs
  git clone https://github.com/intel/ros_opencl_caffe
  cd ~/catkin_ws/
  catkin_make
  catkin_make install
  source install/setup.bash
```
Copy object label file to clCaffe installation location
```Shell
  cp ~/catkin_ws/src/ros_opencl_caffe/opencl_caffe/resources/voc.txt /opt/clCaffe/data/yolo/
```

## Running the Demo
### Inference
  1. Standard USB camera
```Shell
    roslaunch opencl_caffe_launch usb_cam_viewer.launch
```
  2. Intel&reg; RealSense&trade; camera
```Shell
    roslaunch opencl_caffe_launch realsense_viewer.launch
```
  3. Microsoft&reg; Kinect&trade; camera
```Shell
    roslaunch opencl_caffe_launch kinect_viewer.launch
```

### Service
```Shell
  roslaunch opencl_caffe_launch opencl_caffe_srv.launch
```

## Test
Use `rostest` for tests
```Shell
  source devel/setup.bash
  rostest opencl_caffe service.test
  rostest opencl_caffe detector.test
```

## Known Issues
* Only image messages supported in service demo
* Only test on RealSense D400 series camera, Microsoft Kinect v1 camera and Microsoft HD-300 USB camera

## TODO
* Support latest clCaffe

###### *Any security issue should be reported using process at https://01.org/security*
