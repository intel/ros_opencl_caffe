# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.3)
project(opencl_caffe)

find_package(catkin REQUIRED COMPONENTS
  roslib
  roscpp
  sensor_msgs
  std_msgs
  object_msgs
  message_generation
  cv_bridge
  roslint
  pluginlib
  nodelet
)
find_package(OpenCV REQUIRED)

#Ignore sign compare warnings in clCaffe
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

# Flags
if(UNIX OR APPLE)
  # Linker flags.
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # GCC specific flags. ICC is compatible with them.
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # In Clang, -z flags are not compatible, they need to be passed to linker via -Wl.
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  endif()

  # Compiler flags.
  if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    # GCC specific flags.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector")
    endif()
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # Clang is compatbile with some of the flags.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    # Same as above, with exception that ICC compilation crashes with -fPIE option, even
    # though it uses -pie linker option that require -fPIE during compilation. Checksec
    # shows that it generates correct PIE anyway if only -pie is provided.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
  endif()

  # Generic flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names -Wformat -Wformat-security -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  # Dot not forward c++11 flag to GPU beucause it is not supported
  set( CUDA_PROPAGATE_HOST_FLAGS OFF )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
endif()

add_service_files(
   FILES
   Inference.srv
 )

generate_messages(
  DEPENDENCIES
  object_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES opencl_caffe
  CATKIN_DEPENDS roslib roscpp sensor_msgs std_msgs object_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  /opt/clCaffe/include
  /opt/clCaffe/build/include
)

link_directories(
  /opt/clCaffe/build/lib
)

add_executable(opencl_caffe_srv
  src/srv.cpp
  src/detector_gpu.cpp
)

add_library(opencl_caffe_nodelet
  src/nodelet.cpp
  src/detector_gpu.cpp
)

add_dependencies(opencl_caffe_srv ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(opencl_caffe_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(opencl_caffe_srv
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  caffe
  glog
)

target_link_libraries(opencl_caffe_nodelet
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  caffe
)

install(TARGETS opencl_caffe_srv opencl_caffe_nodelet
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY resources
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES nodelet_plugins.xml
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  # static code check
  find_package(roslint REQUIRED)
  roslint_cpp()
  roslint_add_test()

  # run_tests
  # catkin_add_gtest(unittest_detector tests/unittest_detector.cpp)
  # target_link_libraries(unittest_detector ${catkin_LIBRARIES} opencl_caffe_nodelet pthread ${GTEST_LIBRARIES} ${UNITEST_LIBRARIES})
  # catkin_add_gtest(unittest_srv tests/unittest_srv.cpp)
  # target_link_libraries(unittest_srv ${catkin_LIBRARIES} opencl_caffe_srv pthread ${GTEST_LIBRARIES} ${UNITEST_LIBRARIES})

  # rostest
  find_package(rostest REQUIRED)
  add_executable(unittest_detector tests/unittest_detector.cpp)
  target_link_libraries(unittest_detector ${catkin_LIBRARIES} opencl_caffe_nodelet pthread ${GTEST_LIBRARIES} ${UNITEST_LIBRARIES})
  add_executable(unittest_srv tests/unittest_srv.cpp)
  target_link_libraries(unittest_srv ${catkin_LIBRARIES} pthread ${GTEST_LIBRARIES} ${UNITEST_LIBRARIES})
  add_rostest(tests/detector.test)
  add_rostest(tests/service.test)
endif()
