cmake_minimum_required(VERSION 3.22.1)
project(QirPassRunner
  VERSION 0.3
  DESCRIPTION "QIR Pass Runner")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find the Threads package
find_package(Threads REQUIRED)

# Set LLVM_ENABLE_THREADS to ON
set(LLVM_ENABLE_THREADS ON)

# Find LLVMConfig.cmake
find_path(LLVM_CONFIG_DIR LLVMConfig.cmake
  HINTS
  "${CMAKE_PREFIX_PATH}"
  ENV LLVM_DIR
)

if (NOT LLVM_CONFIG_DIR)
  message(FATAL_ERROR "LLVMConfig.cmake not found. Set CMAKE_PREFIX_PATH or LLVM_DIR.")
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_CONFIG_DIR}")

# Include LLVMConfig.cmake
list(APPEND CMAKE_PREFIX_PATH "${LLVM_CONFIG_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_CONFIG_DIR}")
find_package(LLVM REQUIRED CONFIG)

# Set up LLVM-specific flags
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# Add the passes directory to include paths
include_directories(src/headers)

# Specify the path to QDMI using the CUSTOM_QDMI_PATH variable
set(QDMI_PATH ${CUSTOM_QDMI_PATH})

# Specify the path to the directory containing libqdmi.so
link_directories(${CUSTOM_QDMI_PATH}/build)

# Include the directory containing qdmi.hpp
include_directories(${CUSTOM_QDMI_PATH})

# Build QirPassRunner as a shared library
add_library(QirPassRunner SHARED src/QirPassRunner.cpp)

# Link any necessary libraries (e.g., LLVM and QDMI)
target_link_libraries(QirPassRunner PRIVATE LLVM)
target_link_libraries(QirPassRunner PRIVATE qdmi)

# Specify the output directory for the shared library
set_target_properties(QirPassRunner PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/passes")

# Get the executable name from the CUSTOM_EXECUTABLE_NAME variable
set(EXECUTABLE_NAME ${CUSTOM_EXECUTABLE_NAME})

# Build the executable
add_executable(${EXECUTABLE_NAME} src/qpassrunner_d.cpp)

# Link the executable with LLVM and Threads libraries
target_link_libraries(${EXECUTABLE_NAME} PRIVATE LLVM)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Threads::Threads)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE QirPassRunner)

# Include passes subdirectory
add_subdirectory(src/passes)

## Add the client executable
#add_executable(clients/client client.cpp)
#
## Link the client executable with QirPassRunner
#target_link_libraries(clients/client PRIVATE QirPassRunner)
#
## Create a custom target for running clients
#add_custom_target(run_clients
#  COMMAND clients/client
#  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#)

