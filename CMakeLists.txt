cmake_minimum_required(VERSION 3.10)
project(ETH3DMultiViewEvaluation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Option: prefer static libraries so the binary can run without installed dependencies
option(STATIC_LINK "Prefer static libraries and bundle the C/C++ runtime" OFF)

if(STATIC_LINK)
  # Prefer .a archives over shared .so/.dylib when both are available
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".so" ".dylib")
  set(Boost_USE_STATIC_LIBS ON)
  # Bundle libgcc and libstdc++ into the binary
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
  endif()
endif()

# External dependency: boost filesystem
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED COMPONENTS filesystem system)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# External dependency: Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# External dependency: PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Settings.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options(-O2 -msse2 -msse3)
endif()

include_directories(
    ./src
)

# Evaluation executable.
add_executable(ETH3DMultiViewEvaluation
    src/accuracy.cc
    src/accuracy.h
    src/completeness.cc
    src/completeness.h
    src/main.cc
    src/meshlab_project.cc
    src/meshlab_project.h
    src/tinyxml2.cpp
    src/tinyxml2.h
    src/util.h
)
target_link_libraries(ETH3DMultiViewEvaluation
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
)
