cmake_minimum_required(VERSION 3.22.1)
project(QuantumResourceManager
  VERSION 0.3
  DESCRIPTION "Quantum Resource Manager"
)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set the installation directory for the daemon
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX $ENV{HOME} CACHE PATH "Installation directory" FORCE)
endif()
set(INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

# Create the daemon_d.log file
file(WRITE ${CMAKE_INSTALL_PREFIX}/logs/daemon_d.log "")

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
include_directories(
    src/pass_runner/headers 
    src/pass_runner
    src/selector_runner 
    src/scheduler_runner 
    ${CUSTOM_QDMI_PATH})

# Specify the path to the directory containing libqdmi.so
link_directories(${CUSTOM_QDMI_PATH}/build)

# Include the directory containing qdmi.hpp
include_directories(${CUSTOM_QDMI_PATH})

# Build shared libraries
add_library(QirPassRunner SHARED 
    src/pass_runner/QirPassRunner.cpp
    src/pass_runner/PassRunner.cpp)
add_library(PassRunner SHARED 
    src/pass_runner/PassRunner.cpp)
add_library(SelectorRunner SHARED 
    src/selector_runner/SelectorRunner.cpp)
add_library(SchedulerRunner SHARED 
    src/scheduler_runner/SchedulerRunner.cpp)

# Link any necessary libraries (e.g., LLVM and QDMI)
target_link_libraries(QirPassRunner   PRIVATE LLVM)
target_link_libraries(QirPassRunner   PRIVATE QDMI)
target_link_libraries(SchedulerRunner PRIVATE LLVM)
target_link_libraries(SchedulerRunner PRIVATE QDMI)

# Specify the output directory for the shared libraries
set_target_properties(QirPassRunner   
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/pass_runner/passes"
)

set_target_properties(PassRunner
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/pass_runner"
)

set_target_properties(SelectorRunner  
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/selector_runner"
)

set_target_properties(SchedulerRunner 
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/scheduler_runner"
)

set(EXECUTABLE_NAME "daemon_d" CACHE STRING "" FORCE)

# Build the executable
set(SOURCES
    src/daemon_d.cpp
    src/connection_handling.cpp
)

add_executable(${EXECUTABLE_NAME} ${SOURCES})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/src/pass_runner;${CMAKE_INSTALL_PREFIX}/bin/src/pass_runner/passes;${CMAKE_INSTALL_PREFIX}/bin/src/selector_runner;${CMAKE_INSTALL_PREFIX}/bin/src/selector_runner/selectors;${CMAKE_INSTALL_PREFIX}/bin/src/scheduler_runner;${CMAKE_INSTALL_PREFIX}/bin/src/scheduler_runner/schedulers"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Link the executable with LLVM and Threads libraries
target_link_libraries(${EXECUTABLE_NAME} PRIVATE LLVM)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE Threads::Threads)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE QirPassRunner)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE PassRunner)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE SchedulerRunner)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE SelectorRunner)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE rabbitmq)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE QDMI)

# Include libraries subdirectories
add_subdirectory(src/pass_runner/passes)
add_subdirectory(src/selector_runner/selectors)
add_subdirectory(src/scheduler_runner/schedulers)

# Include RabbitMQ header files
target_include_directories(${EXECUTABLE_NAME} PRIVATE /usr/local/include)

# Install the Quantum Resource Manager
install(TARGETS ${EXECUTABLE_NAME} 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
install(TARGETS QirPassRunner
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/src/pass_runner)
install(TARGETS PassRunner 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/src/pass_runner)
install(TARGETS SchedulerRunner 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/src/scheduler_runner)
install(TARGETS SelectorRunner 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/src/selector_runner)

# Uninstall everything
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
  IMMEDIATE @ONLY
)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

