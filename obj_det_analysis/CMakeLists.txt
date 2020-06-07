# ----------------------------------------------------------------------------
#  Root CMake file for ML Project
# ----------------------------------------------------------------------------
#
# Example CMake build line for windows & VS2015
# windows w/cuda: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux w/o cuda: cmake -DUSE_AVX_INSTRUCTIONS=ON -DDLIB_USE_CUDA=OFF ..
#
#
# To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#


cmake_minimum_required(VERSION 2.8.12)

PROJECT(ODA_Project)
set(exe_name "oda_ex")

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
if("${PLATFORM}" STREQUAL "")
    set(PLATFORM "NOT_SET")
    message("The PLATFORM Environment variable is not set on this system.  Setting it to:")
endif()
message("Platform: " ${PLATFORM})

# find the root of the main project
string(FIND ${PROJECT_SOURCE_DIR} "/" INDEX REVERSE)
string(SUBSTRING ${PROJECT_SOURCE_DIR} 0 ${INDEX} PR_ROOT)

option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  ON)

if(WIN32)
    if(${PLATFORM} STREQUAL "Laptop")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.19/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "LaptopN")
        set(COMMON_ROOT "C:/Projects/Common")
        set(DLIB_CONTRIB "C:/Projects/dlib-contrib")
        include(C:/dlib-19.19/dlib/cmake)    
    elseif(${PLATFORM} STREQUAL "BEAST")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.19/dlib/cmake)
    else()
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        include(D:/dlib-19.19/dlib/cmake)
    endif()    
else()
    if(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.19/dlib/cmake)
    else()
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        include(~/dlib-19.19/dlib/cmake)
    endif()
endif()

# grab all of the *.cpp files in the src directory
file(GLOB SOURCES "src/*.cpp" )	

set(project_includes "include" 
    "${PR_ROOT}/common/include"
	"${COMMON_ROOT}/include"
    "${DLIB_CONTRIB}/include" 
	)

# find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})

else()
	message("OpenCV not found, so we won't build the project.")
endif()

message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")

message(" ")

message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib ${OpenCV_LIBS})

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()
endif()

message("Build Complete!")
message(" ")
