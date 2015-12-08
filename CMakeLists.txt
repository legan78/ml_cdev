cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)

project( ML_CDEV )

include_directories(include  )

add_library(gaussianMM  STATIC 	src/gaussianMM.cpp) 
add_library(ModelParams	 STATIC src/ModelParams.cpp)  
add_library(mxTypeModel	 STATIC src/mxTypeModel.cpp)  
add_library(sphericalKMeans	 STATIC src/sphericalKMeans.cpp)
add_library(vonMisesMM 	 STATIC src/vonMisesMM.cpp)



