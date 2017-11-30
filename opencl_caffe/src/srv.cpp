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

bool Srv::handleService(opencl_caffe::Inference::Request& req, opencl_caffe::Inference::Response& resp)
{
  resp.status = -1;
  sensor_msgs::ImagePtr image = boost::make_shared<sensor_msgs::Image>(req.image);
  if (detector_->runInference(image, resp.objs))
  {
    resp.status = 0;
  }
  else
  {
    ROS_ERROR("Inference failed.");
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
