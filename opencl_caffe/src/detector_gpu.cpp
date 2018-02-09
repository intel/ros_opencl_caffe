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
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include "opencl_caffe/detector_gpu.h"

namespace opencl_caffe
{
DetectorGpu::DetectorGpu() : is_fp16_support(false)
{
}
cv::Mat DetectorGpu::resizeImage(const cv::Mat& image)
{
  cv::Mat image_resized;
  if (image.size() == input_size_)
  {
    return image;
  }

  if (image.cols != image.rows)
  {
    if ((input_size_.width * 1.0 / image.cols) < (input_size_.height * 1.0 / image.rows))
    {
      input_size_resized_.width = input_size_.width;
      input_size_resized_.height = (image.rows * input_size_.width) / image.cols;
    }
    else
    {
      input_size_resized_.height = input_size_.height;
      input_size_resized_.width = (image.cols * input_size_.height) / image.rows;
    }
  }
  cv::resize(image, image_resized, input_size_resized_);
  return image_resized;
}

template <typename Dtype>
void DetectorGpu::initInputBlob(const cv::Mat& image, Dtype& input_channels_)
{
  int dx = 0, dy = 0;

  if (input_size_resized_ != input_size_ && input_size_resized_.width != input_size_resized_.height)
  {
    dx = (input_size_.width - input_size_resized_.width) / 2;
    dy = (input_size_.height - input_size_resized_.height) / 2;
  }

  if (dx != 0 || dy != 0)
  {
    for (int i = 0; i < num_channels_; ++i)
    {
      for (int pos = 0; pos < input_size_.width * input_size_.height; ++pos)
      {
        input_channels_[i][pos] = 0.5;
      }
    }
  }

  cv::Mat image_converted;
  image.convertTo(image_converted, num_channels_ == 3 ? CV_32FC3 : CV_32FC1);

  image_converted = image_converted / 255.0;

  for (int i = 0; i < input_size_resized_.height; ++i)
  {
    for (int j = 0; j < input_size_resized_.width; ++j)
    {
      int pos = (i + dy) * input_size_.width + j + dx;
      if (num_channels_ == 3)
      {
        cv::Vec3f pixel = image_converted.at<cv::Vec3f>(i, j);
        input_channels_[0][pos] = pixel.val[2];
        input_channels_[1][pos] = pixel.val[1];
        input_channels_[2][pos] = pixel.val[0];
      }
      else
      {
        cv::Scalar pixel = image_converted.at<float>(i, j);
        input_channels_[0][pos] = pixel.val[0];
      }
    }
  }
  if (dx == 0 && dy == 0)
  {
    input_size_resized_.width = 0;
  }
}

template <typename Dtype>
int DetectorGpu::initNetwork(std::shared_ptr<caffe::Net<Dtype>>& net, std::vector<Dtype*>& input_channels,
                             const std::string& net_cfg, const std::string& weights, const std::string& labels)
{
  net.reset(new caffe::Net<Dtype>(net_cfg, caffe::TEST, caffe::Caffe::GetDefaultDevice()));
  net->CopyTrainedLayersFrom(weights);

  caffe::Blob<Dtype>* input_layer = net->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_size_ = cv::Size(input_layer->width(), input_layer->height());

  input_layer->Reshape(1, num_channels_, input_size_.height, input_size_.width);
  net->Reshape();

  Dtype* input_data = input_layer->mutable_cpu_data();
  int w = input_layer->width();
  int h = input_layer->height();
  for (int i = 0; i < input_layer->channels(); ++i)
  {
    input_channels.push_back(input_data);
    input_data += w * h;
  }

  cv::Mat image = cv::imread(ros::package::getPath("opencl_caffe") + "/resources/cat.jpg");
  initInputBlob(resizeImage(image), input_channels);
  net->Forward();
  return true;
}

int DetectorGpu::loadResources(const std::string& net_cfg, const std::string& weights, const std::string& labels)
{
  if (!boost::filesystem::exists(net_cfg) || !boost::filesystem::exists(weights) || !boost::filesystem::exists(labels))
  {
    ROS_ERROR("Network configuration file or weights file not found!");
    return false;
  }

  std::ifstream fs(labels);
  std::string label_name;
  while (getline(fs, label_name))
  {
    labels_list.push_back(label_name);
  }

  int count = caffe::Caffe::EnumerateDevices(true);
  if (count > 0)
  {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    caffe::device* dev = caffe::Caffe::GetDefaultDevice();
    if (dev->CheckCapability("cl_intel_subgroups") && dev->CheckCapability("cl_intel_subgroups_short"))
    {
      ROS_INFO("FP16 is supported, use FP16.");
      is_fp16_support = true;
      initNetwork(net_fp16_, input_channels_fp16_, net_cfg, weights, labels);
    }
    else
    {
      ROS_INFO("FP16 is NOT supported, use FP32.");
      is_fp16_support = false;
      initNetwork(net_fp32_, input_channels_fp32_, net_cfg, weights, labels);
    }
  }
  else
  {
    ROS_ERROR("GPU was not found or not supported.");
    return false;
  }

  ROS_INFO("Load resources completely!");
  return true;
}

template <typename Dtype>
int DetectorGpu::inference(std::shared_ptr<caffe::Net<Dtype>>& net, std::vector<Dtype*>& input_channels,
                           const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects)
{
  try
  {
    cv::Mat image;

    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();
    cv::cvtColor(cv_bridge::toCvShare(image_msg, "rgb8")->image, image, cv::COLOR_RGB2BGR);
    initInputBlob(resizeImage(image), input_channels);
    net->Forward();
    boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration msdiff = end - start;

    caffe::Blob<Dtype>* result_blob = net->output_blobs()[0];
    const Dtype* result = result_blob->cpu_data();
    const int num_det = result_blob->height();

    for (int k = 0; k < num_det * 7; k += 7)
    {
      int classid = static_cast<int>(result[k + 1]);
      float confidence = result[k + 2];
      int left = 0, right, top = 0, bot;
      if (input_size_resized_.width == 0)
      {
        left = static_cast<int>((result[k + 3] - result[k + 5] / 2.0) * image.cols);
        right = static_cast<int>((result[k + 3] + result[k + 5] / 2.0) * image.cols);
        top = static_cast<int>((result[k + 4] - result[k + 6] / 2.0) * image.rows);
        bot = static_cast<int>((result[k + 4] + result[k + 6] / 2.0) * image.rows);
      }
      else
      {
        left = image.cols * (result[k + 3] - (input_size_.width - input_size_resized_.width) / 2. / input_size_.width) *
               input_size_.width / input_size_resized_.width;
        top = image.rows *
              (result[k + 4] - (input_size_.height - input_size_resized_.height) / 2. / input_size_.height) *
              input_size_.height / input_size_resized_.height;
        float boxw = result[k + 5] * image.cols * input_size_.width / input_size_resized_.width;
        float boxh = result[k + 6] * image.rows * input_size_.height / input_size_resized_.height;
        left -= static_cast<int>(boxw / 2);
        top -= static_cast<int>(boxh / 2);
        right = static_cast<int>(left + boxw);
        bot = static_cast<int>(top + boxh);
      }
      (left < 0) ? left = 0 : left;
      (right > image.cols - 1) ? right = image.cols - 1 : right;
      (top < 0) ? top = 0 : top;
      (bot > image.rows - 1) ? bot = image.rows - 1 : bot;

      object_msgs::ObjectInBox object_in_box;
      object_in_box.object.object_name = labels_list[classid];
      object_in_box.object.probability = confidence;
      object_in_box.roi.x_offset = left;
      object_in_box.roi.y_offset = top;
      object_in_box.roi.height = bot - top;
      object_in_box.roi.width = right - left;
      objects.objects_vector.push_back(object_in_box);
    }

    objects.header = image_msg->header;
    objects.inference_time_ms = msdiff.total_milliseconds();
    return true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not covert from '%s' to 'rgb8'.", image_msg->encoding.c_str());
    return false;
  }
}

int DetectorGpu::runInference(const sensor_msgs::ImagePtr image_msg, object_msgs::ObjectsInBoxes& objects)
{
  return is_fp16_support ? inference(net_fp16_, input_channels_fp16_, image_msg, objects) :
                           inference(net_fp32_, input_channels_fp32_, image_msg, objects);
}
}  // namespace opencl_caffe
