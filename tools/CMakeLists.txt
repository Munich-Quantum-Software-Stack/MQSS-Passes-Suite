cmake_minimum_required(VERSION 3.10)
project(ProjectTests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_FLAGS "-w -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED program_options)
message(STATUS "Boost version: ${Boost_VERSION}")
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost library not found.")
endif()

include_directories(${MLIR_INCLUDE_DIRS})

# Define the common libraries to link
set(COMMON_LIBRARIES
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-spin.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-common.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-nlopt.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-platform-default.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-em-default.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-ensmallen.so
  ${CUDAQ_MLIR_RUNTIME_PATH}/libcudaq-mlir-runtime.so
  pthread)

# quake-to-tikz executable
add_executable(quake-to-tikz quake-to-tikz.cpp)
target_link_libraries(quake-to-tikz PRIVATE 
                        ${COMMON_LIBRARIES} 
                        MQSSPasses
                        Boost::program_options)
target_include_directories(quake-to-tikz PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAQ_MLIR_RUNTIME_PATH})

