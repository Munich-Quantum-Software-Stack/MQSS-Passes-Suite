cmake_minimum_required(VERSION 3.22.1)
project(QirPassSelector
  VERSION 0.1
  DESCRIPTION "QIR Pass Selector Runner")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find the Threads package
find_package(Threads REQUIRED)

# Get the executable name from the CUSTOM_EXECUTABLE_NAME variable
set(EXECUTABLE_NAME ${CUSTOM_EXECUTABLE_NAME})

# Build the executable
add_executable(${EXECUTABLE_NAME} qselectorrunner_d.cpp)

# Include selectors subdirectory
add_subdirectory(selectors)

