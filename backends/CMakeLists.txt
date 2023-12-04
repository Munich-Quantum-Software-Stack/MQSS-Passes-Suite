cmake_minimum_required(VERSION 3.22.1)
project(
  Backends
  VERSION 0.0
  DESCRIPTION "Available backends")

set(CMAKE_CXX_STANDARD 14)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib/backends")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Define the library add_library(Backends SHARED "")
add_library(Backends SHARED "")

# Set the installation directory for the daemon
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      $ENV{HOME}
      CACHE PATH "Installation directory" FORCE)
endif()

# Set LLVM_ENABLE_THREADS to ON
set(LLVM_ENABLE_THREADS ON)

# Find LLVMConfig.cmake
find_path(LLVM_CONFIG_DIR LLVMConfig.cmake HINTS "${CMAKE_PREFIX_PATH}" ENV
                                                 LLVM_DIR)

if(NOT LLVM_CONFIG_DIR)
  message(
    FATAL_ERROR "LLVMConfig.cmake not found. Set CMAKE_PREFIX_PATH or LLVM_DIR."
  )
endif()

# Include LLVMConfig.cmake
list(APPEND CMAKE_PREFIX_PATH "${LLVM_CONFIG_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_CONFIG_DIR}")
find_package(LLVM REQUIRED CONFIG)

# Set up LLVM-specific flags
add_definitions(${LLVM_DEFINITIONS})

# Define the source files for all backends
target_sources(Backends PRIVATE IQMBackend.cpp Q5Backend.cpp Q20Backend.cpp)

# Specify the include directories for Backends library
target_include_directories(
  Backends PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../src/qdmi)

# Link the main executable to the backend library
# target_link_libraries(${EXECUTABLE_NAME} PRIVATE Backends)
target_include_directories(Backends PRIVATE ${LLVM_INCLUDE_DIRS})

message(
  STATUS
    "Installing libBackends.so in: ${CMAKE_INSTALL_PREFIX}/bin/lib/backends")

install(TARGETS Backends
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/lib/backends")
