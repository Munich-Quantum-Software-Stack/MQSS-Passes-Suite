cmake_minimum_required(VERSION 3.22.1)
project(
  qdmi
  VERSION 0.0
  DESCRIPTION "Quantum Device Management Interface")

set(CMAKE_CXX_STANDARD 14)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib/qdmi")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Define the library
add_library(qdmi MODULE qdmi.cpp)

# Set include directories
target_include_directories(qdmi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Installing libqdmi.so in: ${CMAKE_INSTALL_PREFIX}/bin/lib/qdmi")

# Set installation rules
install(TARGETS qdmi LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/lib/qdmi")
