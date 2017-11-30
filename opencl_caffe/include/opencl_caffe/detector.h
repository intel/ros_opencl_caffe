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

#ifndef OPENCL_CAFFE_DETECTOR_H
#define OPENCL_CAFFE_DETECTOR_H

#include <string>
#include <image_transport/image_transport.h>
#include <object_msgs/ObjectsInBoxes.h>

namespace opencl_caffe
{
/** @class Detector
 * @brief Base class for detecting.
 * This class define a common interface of nueral network inference.
 * 1. Load all resources need by a network
 * 2. Run inference
 */
class Detector
{
public:
  /** Default destructor */
  virtual ~Detector() = default;
  /**
   * Load resources from file, construct a caffe Net object.
   *
   * @param[in]   net_cfg   Network configuration file path
   * @param[in]   weights   Neural network weights file path
   * @param[in]   labels    File path of labels of network output classes
   * @return     Status of load resources, true for success or false for failed
   */
  virtual int loadResources(const std::string& net_cfg, const std::string& weights, const std::string& labels) = 0;
  /**
   * Public interface of running inference to infer all objects in image.
   *
   * @param[in]   image_msg   image message subscribed from camera
   * @param[out]  objects     objects inferred
   * @return    Status of run inference, true for success or false for failed
   */
  virtual int runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects) = 0;
};
}  // namespace opencl_caffe

#endif  // OPENCL_CAFFE_DETECTOR_H
