# ML_CDEV
#
#
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)

project( ML_CDEV )

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()


include_directories(include  )

find_package( OpenCV REQUIRED  )

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake) 

find_package(GSL REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )


add_library(dbgLogger STATIC  ./src/dbgLogger.cpp)



add_library(distribution STATIC ./src/distribution.cpp)
target_link_libraries(distribution ${OpenCV_LIBS}) 

add_library(gaussianMM  STATIC 	src/gaussianMM.cpp)

add_library(ModelParams	 STATIC src/ModelParams.cpp)  
add_library(mxTypeModel	 STATIC src/mxTypeModel.cpp)  
add_library(sphericalKMeans	 STATIC src/sphericalKMeans.cpp)
add_library(vonMisesMM 	 STATIC src/vonMisesMM.cpp)


