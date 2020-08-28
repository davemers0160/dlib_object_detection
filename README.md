# Object Detection Repository using Dlib
This repository is a combination of several object detection projects based on the dlib library.

## Dependencies

The code in this repository has the following dependecies

1. [CMake 2.8.12+](https://cmake.org/download/ )
2. [davemers0160 common code repository](https://github.com/davemers0160/Common )
3. [davemers0160 dlib-contrib repository](https://github.com/davemers0160/dlib-contrib )
4. [dlib library](http://dlib.net/ )
5. [OpenCV v4+](https://opencv.org/releases/ )

## Repository Breakdown

### Trainers:

These sub-projects are dedicated to training dnn's for object detection and classification.

### obj_det_trainer

This folder contains the project code that runs the training of a given network against a given dataset using the array style input for each color channel.

### obj_det_rgb_trainer

This folder contains the project code that runs the training of a given network against a given dataset using the rgb pixel color input.

### Analysis:
These sub-projects are dedicated to analyzing a trained dnn's performance for a given dataset.

### obj_det_analysis

This folder contains the project code that runs the performance analysis of a given network against a given dataset using the array style input for each color channel.

### obj_det_rgb_analysis

This folder contains the project code that runs the performance analysis of a given network against a given dataset using the rgb pixel color input.

### Dynamic Library generation
These sub-projects are designed to compile a trained network into a shared library that can be used by other programs.

### obj_det_lib

This folder contains the project code that compiles a dnn object detector into a shared library that can be used by other programs.  The network input uses the array style input for each color channel.

### Dynamic Library Linking Examples
These sub-projects provide examples on how to link the compiled shared library to other C++ code.

### obj_det_run

This folder contains the project code that tests the linking in C++ to the shared object detector library at compile time.

### obj_det_run_lib

This folder contains the project code that allows you supply the compiled shared object detector library at runtime.

### Network Result Viewer

### obj_det_viewer

This folder contains the python code that links the C++ shared object detector library and provides a web interface through a bokeh server that allows you to view the object detection results and look at the final heatmap results. This viewer is designed to work with the arrays style input network architecture.

### common

This folder contains the project code that is shared between all of the projects.
