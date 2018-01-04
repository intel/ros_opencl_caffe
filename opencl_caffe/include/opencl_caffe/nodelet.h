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

#ifndef OPENCL_CAFFE_NODELET_H
#define OPENCL_CAFFE_NODELET_H

#include <string>
#include <nodelet/nodelet.h>
#include "opencl_caffe/detector.h"

namespace opencl_caffe
{
/** @class Nodelet
 * @brief Nodelet of the detector
 * This class implement a nodelet for the detector.
 * 1. Load all resources need by the detector
 * 2. Subscribe the image messages.
 */
class Nodelet : public nodelet::Nodelet
{
private:
  /** Shared pointor of a detector */
  std::shared_ptr<Detector> detector_;
  /** Subscriber for subscribing the image messages*/
  ros::Subscriber sub_;
  /** Publisher for publishing the objects information messages */
  ros::Publisher pub_;

public:
  /** Nodelet onInit  function */
  virtual void onInit();
  /**
   * Callback method. Called when image message comes.
   *
   * @param[in] image_msg Image message
   */
  void cbImage(const sensor_msgs::ImagePtr image_msg);
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

#endif  // OPENCL_CAFFE_NODELET_H
