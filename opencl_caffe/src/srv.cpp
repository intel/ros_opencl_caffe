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
#include <cv_bridge/cv_bridge.h>
#include <pluginlib/class_list_macros.h>
#include "opencl_caffe/detector_gpu.h"
#include "opencl_caffe/srv.h"

namespace opencl_caffe
{
Srv::Srv(ros::NodeHandle& n)
{
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

  loadResources(net_config_path, weights_path, labels_path);
  service_ = n.advertiseService("run_inference", &Srv::handleService, this);
}

bool Srv::handleService(object_msgs::DetectObject::Request& req, object_msgs::DetectObject::Response& resp)
{
  for (auto image_path : req.image_paths)
  {
    cv_bridge::CvImage cv_image;
    sensor_msgs::Image ros_image;
    cv_image.image = cv::imread(image_path);
    cv_image.encoding = "bgr8";
    cv_image.toImageMsg(ros_image);

    object_msgs::ObjectsInBoxes objects;

    if (!detector_->runInference(boost::make_shared<sensor_msgs::Image>(ros_image), objects))
    {
      ROS_ERROR("Detect object failed.");
      return false;
    }

    resp.objects.push_back(objects);
  }
  
  return true;
}

void Srv::loadResources(const std::string net_config_path, const std::string weights_path,
                        const std::string labels_path)
{
  detector_.reset(new DetectorGpu());
  if (!detector_->loadResources(net_config_path, weights_path, labels_path))
  {
    ROS_FATAL("Load resource failed.");
    ros::shutdown();
  }
}
}  // namespace opencl_caffe

int main(int argc, char** argv)
{
  ros::init(argc, argv, "opencl_caffe");

  ros::NodeHandle n("~");
  opencl_caffe::Srv srv(n);

  ros::spin();

  return 0;
}
