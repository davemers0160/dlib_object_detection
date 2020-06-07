# Object Detector Library Code

This project is designed to compile a dlib object detection network into a dynamically linked library (*.dll/*.so) so that is can be used byt other code/projects.

## Dependencies

The code has the following dependecies

1. [CMake 2.8.12+](https://cmake.org/download/ )
2. [davemers0160 common code repository](https://github.com/davemers0160/Common )
3. [davemers0160 dlib-contrib repository](https://github.com/davemers0160/dlib-contrib )
4. [dlib library](http://dlib.net/ )
5. [OpenCV v4+](https://opencv.org/releases/ )

Follow the instructions for each of the dependencies according to your operating system.  

For dlib it should be compiled as a dynamic library.  Starting from the dlib folder within the entire dlib library:

```
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=1 ..
make
sudo make install
sudo ldconfig
```

This will build the dlib.so files and add a cmake package that can be found by calling within a CMakeLists.txt file:

```
find_package(dlib)
```

## Build

The project uses CMake as the primary mechanism to build the executables.  There are some modifications that may have to be made to the CMakeLists.txt file in order to get the project to build successfully.

The first thing that must be done is to create an environment variable called "PLATFORM".  The CMakeLists.txt file uses this variable to determine where to look for the other required repositories and/or libraries.  These will be machine specific.

To create an environment variable in Windows (drop the -m if you do not have elevated privileges):
```
setx -m PLATFORM MY_PC
```

In Linux (usually placed in .profile or .bashrc):
```
export PLATFORM=MY_PC
```

In the CMakeLists.txt file make sure to add a check for the platform you've added and point to the right locations for the repositories/libraries.

### Windows

From the directory that contains this file, execute the following commands in a Windows command window:

```
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -T host=x64 ..
cmake --build . --config Release
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. 

### Linux

From the directory that contains this file, execute the following commands in a terminal window:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release -- -j4
```

Or you can use the cmake-gui and set the "source code" location to the location of the CmakeLists.txt file and the set the "build" location to the build folder. Then open a terminal window and navigate to the build folder and execute the follokwing command:

```
cmake --build . --config Release -- -j4
```

The -- -j4 tells the make to use 4 cores to build the code.  This number can be set to as many cores as you have on your PC.

## Running

To run the code you must create a secondary project that links to the shared library.  The following two projects use this shared library in their code base:

- [Object Detection Run](https://github.com/davemers0160/dlib_object_detection/tree/master/obj_det_run ) C++ project to use the shared library
- [Object Detection Viewer](https://github.com/davemers0160/dlib_object_detection/tree/master/obj_det_viewer ) Python code that uses the shared library and shows the results using a web browser

