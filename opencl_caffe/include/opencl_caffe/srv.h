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

#ifndef OPENCL_CAFFE_SRV_H
#define OPENCL_CAFFE_SRV_H

#include <string>
#include <opencl_caffe/Inference.h>
#include "opencl_caffe/detector.h"

namespace opencl_caffe
{
/** @class Srv
 * @brief Service of the detector
 * This class implement a service for the detector.
 * 1. Load all resources need by the detector
 * 2. Advertise a service
 */
class Srv
{
private:
  /** Service server for advertising service */
  ros::ServiceServer service_;
  /** Shared pointor of a detector */
  std::shared_ptr<Detector> detector_;

public:
  /**
   * Explicit constructor.
   *
   * @param[in]   n   Service node handle
   */
  explicit Srv(ros::NodeHandle& n);
  /**
   * Callback method.
   *
   * @param[in]   req   Request of service
   * @param[out]  resp  Response of service
   */
  bool handleService(opencl_caffe::Inference::Request& req, opencl_caffe::Inference::Response& resp);
  /**
   * Load resources for detector.
   *
   * @param[in]   net_config_path   Network configuration file path
   * @param[in]   weights_path      Neural network weights file path
   * @param[in]   labels_path       File path of labels of network output classes
   */
  void loadResources(const std::string net_config_path, const std::string weights_path, const std::string labels_path);
};
}  // namespace opencl_caffe

#endif  // OPENCL_CAFFE_SRV_H
