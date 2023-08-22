cmake_minimum_required(VERSION 3.22.1)
project(QirPassManager
  VERSION 0.1
  DESCRIPTION "QIR Pass Manager")

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
link_directories(${LLVM_LIBRARY_DIRS})

# Add the passes directory to include paths
include_directories(src/headers)

# Get the executable name from the CUSTOM_EXECUTABLE_NAME variable
set(EXECUTABLE_NAME ${CUSTOM_EXECUTABLE_NAME})

# Specify the main source file using the CUSTOM_MAIN_SOURCE variable
set(MAIN_SOURCE ${CUSTOM_MAIN_SOURCE})

# Build your executable
add_executable(${EXECUTABLE_NAME} ${MAIN_SOURCE})

# Include passes subdirectory
add_subdirectory(src/passes)

# Link your executable with LLVM libraries
target_link_libraries(${EXECUTABLE_NAME} PRIVATE LLVM)
