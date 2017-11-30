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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <object_msgs/ObjectsInBoxes.h>

void cbTimeSync(const sensor_msgs::ImageConstPtr& img, const object_msgs::ObjectsInBoxes::ConstPtr& objs)
{
  try
  {
    cv::Mat cv_image = cv_bridge::toCvShare(img, "bgr8")->image;
    for (auto obj : objs->objects_vector)
    {
      std::stringstream ss;
      ss << obj.object.object_name << ':' << obj.object.probability;
      cv::rectangle(cv_image, cvPoint(obj.roi.x_offset, obj.roi.y_offset),
                    cvPoint(obj.roi.x_offset + obj.roi.width, obj.roi.y_offset + obj.roi.height),
                    cv::Scalar(255, 242, 35));
      cv::putText(cv_image, ss.str(), cvPoint(obj.roi.x_offset, obj.roi.y_offset + 20), cv::FONT_HERSHEY_PLAIN, 1.0f,
                  cv::Scalar(0, 255, 255));
    }

    cv::imshow("opencl_caffe_viewer", cv_image);
    cv::waitKey(5);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", img->encoding.c_str());
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "opencl_caffe_viewer");
  ros::NodeHandle n;

  message_filters::Subscriber<sensor_msgs::Image> cam_sub(n, "/usb_cam/image_raw", 1);
  message_filters::Subscriber<object_msgs::ObjectsInBoxes> objs_sub(n, "opencl_caffe/inference", 1);

  message_filters::TimeSynchronizer<sensor_msgs::Image, object_msgs::ObjectsInBoxes> ts(cam_sub, objs_sub, 60);
  ts.registerCallback(boost::bind(&cbTimeSync, _1, _2));
  ros::spin();

  return 0;
}
