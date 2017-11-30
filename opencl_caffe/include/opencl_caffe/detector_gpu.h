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

#ifndef OPENCL_CAFFE_DETECTOR_GPU_H
#define OPENCL_CAFFE_DETECTOR_GPU_H

#include <string>
#include <vector>
#include <caffe/caffe.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <object_msgs/ObjectsInBoxes.h>
#include "opencl_caffe/detector.h"

namespace opencl_caffe
{
/** @class DetectorGpu
 * @brief A implamentation of GPU detecting.
 * This class implament a caffe GPU object inference with 16 bit and 32 bit float point.
 * 1. Load resources need by caffe network
 * 2. Initialize the network (template for fp16 and fp32)
 * 3. Pre-process input image
 * 4. Infer the objects in image
 */
class DetectorGpu : public Detector
{
private:
  /** 16 bit caffe network */
  std::shared_ptr<caffe::Net<half>> net_fp16_;
  /** 32 bit caffe network */
  std::shared_ptr<caffe::Net<float>> net_fp32_;
  /** 16 bit input blob pointors of each channel */
  std::vector<half*> input_channels_fp16_;
  /** 32 bit input blob pointors of each channel */
  std::vector<float*> input_channels_fp32_;
  /** Origin input width and height */
  cv::Size input_size_;
  /** Input width and height after resize */
  cv::Size input_size_resized_;
  /** Number of input channels */
  int num_channels_;
  /** The list of label names of object classes */
  std::vector<std::string> labels_list;
  /** Whether the GPU support 16 bit float point */
  bool is_fp16_support;

private:
  /**
   * Initialize the caffe network, template function is for fp16 and fp32.
   *
   * @param[in, out]  net             Caffe network
   * @param[in, out]  input_channels  input blob pointors of each channel
   * @param[in]       net_cfg         Network configuration file path
   * @param[in]       weights         Neural network weights file path
   * @param[in]       labels          File path of labels of network output classes
   * @return    Status of init network, true for success or false for failed
   */
  template <typename Dtype>
  int initNetwork(std::shared_ptr<caffe::Net<Dtype>>& net, std::vector<Dtype*>& input_channels,
                  const std::string& net_cfg, const std::string& weights, const std::string& labels);
  /**
   * Initialize the input blobs for clCaffe
   *
   * @param[in]       image           cv::Mat image input
   * @param[in, out]  input_channels  input blob pointors of each channel
   * @return    No returns
   */
  template <typename Dtype>
  void initInputBlob(const cv::Mat& image, Dtype& input_channels_);
  /**
   * Resize the image first
   *
   * @param[in]       image           cv::Mat image input
   * @return    Resized image with cv::Mat format
   */
  cv::Mat resizeImage(const cv::Mat& image);
  /**
   * Run inference to infer all objects in image.
   *
   * @param[in, out]  net             Caffe network
   * @param[in, out]  input_channels  Input blob pointors of each channel
   * @param[in]       image_msg       Image messages as input
   * @param[out]       objects        Objects inferred as output
   * @return    Status of inference, true for success or false for failed
   */
  template <typename Dtype>
  int inference(std::shared_ptr<caffe::Net<Dtype>>& net, std::vector<Dtype*>& input_channels,
                const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects);

public:
  /** Constructor, intialize default value of is_fp16_support */
  DetectorGpu();
  /** Deafult deconstructor */
  ~DetectorGpu() = default;
  /**
   * Load resources from file, construct a caffe Net object.
   *
   * @param[in]   net_cfg   Network configuration file path
   * @param[in]   weights   Neural network weights file path
   * @param[in]   labels    File path of labels of network output classes
   * @return     Status of load resources, true for success or false for failed
   */
  int loadResources(const std::string& net_cfg, const std::string& weights, const std::string& labels);
  /**
   * Public interface of running inference to infer all objects in image.
   *
   * @param[in]   image_msg   image message subscribed from camera
   * @param[out]  objects     objects inferred
   * @return    Status of run inference, true for success or false for failed
   */
  int runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects);
};
}  // namespace opencl_caffe

#endif  // OPENCL_CAFFE_DETECTOR_GPU_H
