cmake_minimum_required(VERSION 3.22.1)
project(
  FoMaC
  VERSION 0.0
  DESCRIPTION "Figure of Merits and Constraints")

set(CMAKE_CXX_STANDARD 14)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib/fomac")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Define the library
add_library(FoMaC MODULE fomac.cpp)

# Set include directories
target_include_directories(FoMaC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FoMaC PRIVATE ../src/qdmi)

message(
  STATUS "Installing libFoMaC.so in: ${CMAKE_INSTALL_PREFIX}/bin/lib/fomac")

# Set installation rules
install(TARGETS FoMaC
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/lib/fomac")
