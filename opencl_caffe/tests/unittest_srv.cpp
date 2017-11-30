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
#include <opencl_caffe/Inference.h>

std::string matType2Encoding(int mat_type)
{
  switch (mat_type)
  {
    case CV_8UC1:
      return "mono8";
    case CV_8UC3:
      return "bgr8";
    case CV_16SC1:
      return "mono16";
    case CV_8UC4:
      return "rgba8";
    default:
      throw std::runtime_error("Unsupported encoding type");
  }
}

void convertFrameToMessage(const cv::Mat* frame, size_t frame_id, sensor_msgs::Image* image_msg)
{
  image_msg->height = frame->rows;
  image_msg->width = frame->cols;
  image_msg->encoding = matType2Encoding(frame->type());
  image_msg->step = static_cast<sensor_msgs::Image::_step_type>(frame->step);
  size_t size = frame->step * frame->rows;
  image_msg->data.resize(size);
  memcpy(&image_msg->data[0], frame->data, size);
  image_msg->header.frame_id = std::to_string(frame_id);
}

TEST(UnitTestSrv, testSrv)
{
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<opencl_caffe::Inference>("opencl_caffe/opencl_caffe_srv/run_inference");
  opencl_caffe::Inference inf;

  cv::Mat image = cv::imread(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  convertFrameToMessage(&image, 0, &inf.request.image);
  client.waitForExistence(ros::Duration(60));
  ASSERT_TRUE(client.call(inf));
  ASSERT_GE(inf.response.objs.objects_vector.size(), 1);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "opencl_caffe_test");
  return RUN_ALL_TESTS();
}
