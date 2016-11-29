cmake_minimum_required (VERSION 3.6)

project("libOTe")


set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)


#############################################
#            Flag and #defines              #
#############################################
add_definitions(-DSOLUTION_DIR=\"${CMAKE_SOURCE_DIR}/../\")
set(CMAKE_C_FLAGS "-ffunction-sections -Wall   -Wfatal-errors -pthread")
set(CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS}  -std=c++14")

# Set a default build type for single-configuration
# CMake generators if no build type is set.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release")
  message(WARNING "\nConfiguration NOT set to Release. Please call cmake with CMAKE_BUILD_TYPE=Release")
endif()

# Select flags.
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")


#############################################
#              Build Frontend               #
#############################################

include_directories("${CMAKE_SOURCE_DIR}") 
file(GLOB_RECURSE SRC_FRONTEND0 ${CMAKE_SOURCE_DIR}/Common/*.cpp)
file(GLOB_RECURSE SRC_FRONTEND1 ${CMAKE_SOURCE_DIR}/MLTree/*.cpp)
file(GLOB_RECURSE SRC_FRONTEND2 ${CMAKE_SOURCE_DIR}/Util/*.cpp)
file(GLOB SRC_FRONTEND3 ${CMAKE_SOURCE_DIR}/*.cpp)
add_executable(frontend.exe  ${SRC_FRONTEND0} ${SRC_FRONTEND1} ${SRC_FRONTEND2} ${SRC_FRONTEND3})

