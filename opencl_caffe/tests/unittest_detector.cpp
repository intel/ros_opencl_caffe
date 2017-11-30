/*
 * Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gtest/gtest.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include "opencl_caffe/detector_gpu.h"

TEST(UnitTestDetector, testDetectorGPU)
{
  opencl_caffe::DetectorGpu detector;

  ros::NodeHandle n("~");
  std::string net_config_path, weights_path, labels_path;
  if (!n.getParam("net_config_path", net_config_path))
  {
    ROS_WARN("param net_cfg_path not set, use default");
  }
  if (!n.getParam("weights_path", weights_path))
  {
    ROS_WARN("param weights_path not set, use default");
  }
  if (!n.getParam("labels_path", labels_path))
  {
    ROS_WARN("param labels_path not set, use default");
  }

  // use ASSERT instead of EXPECT
  ASSERT_TRUE(boost::filesystem::exists(net_config_path));
  ASSERT_TRUE(boost::filesystem::exists(weights_path));
  ASSERT_TRUE(boost::filesystem::exists(labels_path));

  ASSERT_TRUE(detector.loadResources(net_config_path, weights_path, labels_path));

  cv::Mat image = cv::imread(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
  object_msgs::ObjectsInBoxes obejcts;
  ASSERT_TRUE(detector.runInference(image_msg, obejcts));
  ASSERT_GE(obejcts.objects_vector.size(), 1);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "ropencl_caffe_test");
  return RUN_ALL_TESTS();
}
