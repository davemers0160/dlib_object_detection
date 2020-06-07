# dlib_obj_detector
This repository is a combination of several object detection projects based on the dlib library.

## Dependencies

The code in this repository has the following dependecies

1. [CMake 2.8.12+](https://cmake.org/download/ )
2. [davemers0160 common code repository](https://github.com/davemers0160/Common )
3. [davemers0160 dlib-contrib repository](https://github.com/davemers0160/dlib-contrib )
4. [dlib library](http://dlib.net/ )
5. [OpenCV v4+](https://opencv.org/releases/ )

## Repository Breakdown

### common

This folder contains the project code that is shared between all of the projects.

### obj_det_analysis

This folder contains the project code that runs the performance analysis of a given network against a given dataset.

### obj_det_lib

This folder contains the project code that compiles a dnn object detector into a shared library that can be used by other programs.

### obj_det_run

This folder contains the project code that tests the linking in C++ to the shared object detector library.

### obj_det_trainer

This folder contains the project code that runs the training of a given network against a given dataset.

### obj_det_viewer

This folder contains the python code that links the C++ shared object detector library and provides a web interface through a boken server.

