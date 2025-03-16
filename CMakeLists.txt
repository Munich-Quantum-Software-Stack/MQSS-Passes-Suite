# Project name and language specification
cmake_minimum_required(VERSION 3.19...3.30)
project(
  MLIRPasses
  VERSION 1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_FLAGS "-w -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

# Set default build type to Debug if none is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type (Debug, Release)" FORCE)
endif()

# Define macros for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Configuring for Debug mode")
  add_compile_definitions(DEBUG)
endif()

include(cmake/ExternalDependencies.cmake) # doxygen dependencies

# ############################# CONFIGURATION OF LLVM ##########################
find_package(LLVM 16 CONFIG QUIET)

if(NOT LLVM_DIR)
  message(STATUS "LLVM_DIR not found, will try with llvm-config executable.")

  macro(find_llvm_config name version_major)
    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if(${extra_count} GREATER 0)
      list(GET extra_args 0 path)
      find_program(
        LLVM_CONFIG
        NAMES ${name}
        PATHS ${path}
        NO_DEFAULT_PATH
        DOC "Path to llvm-config tool")
    else()
      find_program(
        LLVM_CONFIG
        NAMES ${name}
        DOC "Path to llvm-config tool")
    endif()

    if(LLVM_CONFIG)
      execute_process(
        COMMAND ${LLVM_CONFIG} --version
        RESULT_VARIABLE LLVM_CONFIG_RESULT
        OUTPUT_VARIABLE LLVM_CONFIG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(NOT LLVM_CONFIG_VERSION MATCHES "^${version_major}[.][0-9]+[.][0-9]+")
        unset(LLVM_CONFIG CACHE)
      endif()
    endif()
  endmacro()

  # First we try to find llvm-config in the llvm submodule.
  find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR}
                   "${CMAKE_CURRENT_SOURCE_DIR}/tpls/llvm/build/bin")

  # Try to find a system llvm-config and make sure it is the correct version.
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR})
  endif()

  # If it is not the correct version, try finding llvm-config-VERSION
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config-${LLVM_VERSION_MAJOR} ${LLVM_VERSION_MAJOR})
    if(LLVM_CONFIG)
      set(NVQPP_LLVM_EXECUTABLE_SUFFIX -${LLVM_VERSION_MAJOR})
    endif()
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG} --cmakedir
    RESULT_VARIABLE LLVM_CONFIG_RESULT
    OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(LLVM_CONFIG_RESULT)
    message(
      FATAL_ERROR "Could not find suitable llvm-config(-${LLVM_VERSION_MAJOR}).\
      \nTry providing valid -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm.")
  else()
    find_package(
      LLVM
      ${LLVM_VERSION_MAJOR}
      REQUIRED
      CONFIG
      HINTS
      ${LLVM_CONFIG_CMAKE_DIR}
      NO_DEFAULT_PATH)
  endif()
endif()

if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_BINARY_DIR}/lib/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG)

if(NOT Clang_DIR)
  set(Clang_DIR ${LLVM_BINARY_DIR}/lib/cmake/clang)
endif()

find_package(Clang CONFIG)
if(NOT Clang_FOUND)
  message(STATUS "Clang not found, turning off C++ Frontend.")
  set(CUDAQ_DISABLE_CPP_FRONTEND ON)
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Add LLVM, Clang and MLIR headers to the include path
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS}
                    ${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")

if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()

set(LLVM_LIT_ARGS
    "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")
# ############### ADDITIONAL PACKAGES #######################################
# Tell where to search for Find<package>.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# set(CMAKE_CXX_STANDARD 11) Find QDMI package find_package(qdmi REQUIRED) Find
# QInfo package find_package(qinfo REQUIRED)
find_package(Z3 REQUIRED)

find_package(mqt-qcec REQUIRED)
# Find the QMAP package
set(CMAKE_CXX_STANDARD 17)
set(MQT_QMAP_TESTS OFF)
find_package(mqt-qmap REQUIRED)
# Restore the original CXX flags and compiler
set(CMAKE_CXX_STANDARD 20)

# ###################### CUDAQ settings ###################################
set(CUDAQ_MAIN_INCLUDE_DIR "${CUDAQ_SOURCE_DIR}/include")

set(CUDAQ_BINARY_DIR "${CUDAQ_SOURCE_DIR}/build")
set(CUDAQ_CMAKE_DIR "${CUDAQ_SOURCE_DIR}/cmake")
set(CUDAQ_INCLUDE_DIR "${CUDAQ_SOURCE_DIR}/build/include")
set(CUDAQ_LIBRARY_DIR "${CUDAQ_SOURCE_DIR}/build/lib")
set(CUDAQ_TOOLS_DIR "${CUDAQ_SOURCE_DIR}/build/bin")
set(CUDAQ_MLIR_RUNTIME_PATH "${CUDAQ_SOURCE_DIR}/build/lib")

include_directories(BEFORE ${CUDAQ_MAIN_INCLUDE_DIR}
                    ${CUDAQ_BINARY_DIR}/include ${CUDAQ_SOURCE_DIR}/runtime)

# Validate CUDAQ dependencies
if(NOT EXISTS ${CUDAQ_BINARY_DIR}/lib/libcudaq-mlir-runtime.so)
  message(
    FATAL_ERROR
      "Invalid CUDAQ dir: ${CUDAQ_SOURCE_DIR}. Cannot find libcudaq.so in build directory."
  )
endif()
# ##############################################################################

set(MQT_QMAP_INCLUDE_DIR "${mqt-qmap_SOURCE_DIR}/include/")
set(MQT_QCEC_INCLUDE_DIR "${mqt-qcec_SOURCE_DIR}/include/")
set(MQT_CORE_INCLUDE_DIR "${mqt-core_SOURCE_DIR}/include/mqt-core")
set(qdmi_INCLUDE_DIRS "${QDMI_INCLUDE_DIRS}/qdmi")
set(qinfo_INCLUDE_DIRS "${qinfo_SOURCE_DIR}/include/qinfo")
message(STATUS "mqt include dir: ${MQT_QMAP_INCLUDE_DIR}")
message(STATUS "QINFO INCLUDE DIRS ${qinfo_INCLUDE_DIRS}")

# Get all variable names in CMake get_cmake_property(all_vars VARIABLES)
#
# Loop through and print each variable and its value foreach(var ${all_vars})
# message(STATUS "${var} = ${${var}}") endforeach()

include_directories(
  BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Passes/
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Support/
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Support/CodeGen/
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Optimizer/
  ${MQT_QMAP_INCLUDE_DIR}
  ${MQT_CORE_INCLUDE_DIR}
  ${qdmi_INCLUDE_DIRS}
  ${qinfo_INCLUDE_DIRS})

# Collect all .cpp files in the src directory
file(GLOB SRC_CODEGEN "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/CodeGen/*.cpp")
file(GLOB SRC_DECOMPOSITIONS
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Decompositions/*.cpp")
file(GLOB SRC_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Examples/*.cpp")
file(GLOB SRC_TRANSFORMS
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes/Transforms/*.cpp")
file(GLOB SRC_SUPPORT_QUAKE
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/Support/CodeGen/Quake/*.cpp")
file(GLOB SRC_OPTIMIZER "${CMAKE_CURRENT_SOURCE_DIR}/lib/Optimizer/*.cpp")

add_mlir_library(
  SupportQuake
  SHARED
  ${SRC_SUPPORT_QUAKE}
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms)
# target_include_directories( SupportQuake PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# $<INSTALL_INTERFACE:include>)

add_mlir_library(
  MLIRPasses
  SHARED
  ${SRC_CODEGEN}
  ${SRC_DECOMPOSITIONS}
  ${SRC_EXAMPLES}
  ${SRC_TRANSFORMS}
  LINK_LIBS
  PUBLIC
  SupportQuake
  MQT::CoreIR
  MQT::CoreNA
  MQT::QMapSC
  MQT::QMapSCHeuristic
  MLIRIR
  MLIRPass
  MLIRTransforms)

add_mlir_library(Optimizer SHARED ${SRC_OPTIMIZER} LINK_LIBS PUBLIC MLIRIR)
# target_include_directories( Optimizer PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Optimizer>
# $<INSTALL_INTERFACE:include>)

target_compile_options(SupportQuake PRIVATE -fexceptions -frtti)
target_compile_options(MLIRPasses PRIVATE -fexceptions -frtti)

# Create an alias target for convenience
add_library(MQSS::MLIRPasses ALIAS MLIRPasses)
add_library(MQSS::SupportQuake ALIAS SupportQuake)
add_library(MQSS::Optimizer ALIAS Optimizer)

# add tests
if(BUILD_MLIR_PASSES_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(tests)
endif()

# add documentation
if(BUILD_MLIR_PASSES_DOCS)
  add_subdirectory(docs)
endif()

if(BUILD_MLIR_PASSES_TOOLS)
  add_subdirectory(tools)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)

# Enable FetchContent for integration
include(FetchContent)
