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
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <object_msgs/DetectObject.h>

TEST(UnitTestSrv, testSrv)
{
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<object_msgs::DetectObject>("opencl_caffe/opencl_caffe_srv/run_inference");
  object_msgs::DetectObject inf;

  inf.request.image_paths.push_back(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  inf.request.image_paths.push_back(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  client.waitForExistence(ros::Duration(120));
  ASSERT_TRUE(client.call(inf));
  ASSERT_EQ(inf.response.objects.size(), 2);
  ASSERT_GE(inf.response.objects[0].objects_vector.size(), 1);
  ASSERT_GE(inf.response.objects[1].objects_vector.size(), 1);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "opencl_caffe_test");
  return RUN_ALL_TESTS();
}
