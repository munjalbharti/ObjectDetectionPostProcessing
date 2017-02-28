# Copyright (c) 2016 Bharti Munjal
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ObjectDetectionPostProcessing C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set (CMAKE_CXX_STANDARD 11)

#==============================================================================
# Dependencies
#==============================================================================

# include Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
#SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
#FIND_PACKAGE(Eigen REQUIRED)
#INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 3.0.0 COMPONENTS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs REQUIRED)

#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O1 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.h")
FILE(GLOB srcs "src/*.cpp")
#FILE(GLOB incs "*.hpp")

ADD_EXECUTABLE(objdetpost ${incs} ${srcs} )
TARGET_LINK_LIBRARIES(objdetpost ${OpenCV_LIBS})

