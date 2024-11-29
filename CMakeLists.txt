cmake_minimum_required(VERSION 3.13)
# Project name and language specification
project(MQSSMLIRCustomPasses LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

############################## CONFIGURATION OF LLVM ##########################
find_package(LLVM 16 CONFIG QUIET)

if(NOT LLVM_DIR)
  message(STATUS "LLVM_DIR not found, will try with llvm-config executable.")

  macro(find_llvm_config name version_major)
    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0)
      list(GET extra_args 0 path)
      find_program(LLVM_CONFIG NAMES ${name} PATHS ${path} NO_DEFAULT_PATH
        DOC "Path to llvm-config tool")
    else()
      find_program(LLVM_CONFIG NAMES ${name} DOC "Path to llvm-config tool")
    endif()

    if(LLVM_CONFIG)
      execute_process(
        COMMAND ${LLVM_CONFIG} --version
        RESULT_VARIABLE LLVM_CONFIG_RESULT
        OUTPUT_VARIABLE LLVM_CONFIG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
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
    if (LLVM_CONFIG)
      set(NVQPP_LLVM_EXECUTABLE_SUFFIX -${LLVM_VERSION_MAJOR})
    endif()
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG} --cmakedir
    RESULT_VARIABLE LLVM_CONFIG_RESULT
    OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(LLVM_CONFIG_RESULT)
    message(FATAL_ERROR
      "Could not find suitable llvm-config(-${LLVM_VERSION_MAJOR}).\
      \nTry providing valid -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm.")
  else()
    find_package(LLVM ${LLVM_VERSION_MAJOR} REQUIRED CONFIG
      HINTS ${LLVM_CONFIG_CMAKE_DIR} NO_DEFAULT_PATH)
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
if (NOT Clang_FOUND)
  message(STATUS "Clang not found, turning off C++ Frontend.")
  set (CUDAQ_DISABLE_CPP_FRONTEND ON)
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
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")

if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()

set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
#################################################################################
# CUDAQ settings

set(CUDAQ_MAIN_INCLUDE_DIR "/workspaces/cuda-quantum/include")
set(CUDAQ_SOURCE_DIR "/workspaces/cuda-quantum")
set(CUDAQ_BINARY_DIR "/workspaces/cuda-quantum/build")
set(CUDAQ_CMAKE_DIR "/workspaces/cuda-quantum/cmake")
set(CUDAQ_INCLUDE_DIR "/workspaces/cuda-quantum/build/include")
set(CUDAQ_LIBRARY_DIR "/workspaces/cuda-quantum/build/lib")
set(CUDAQ_TOOLS_DIR "/workspaces/cuda-quantum/build/bin")
set(CUDAQ_MLIR_RUNTIME_PATH "/workspaces/cuda-quantum/build/lib")

include_directories(BEFORE
         ${CUDAQ_SOURCE_DIR}/include
         ${CUDAQ_BINARY_DIR}/include
#         ${CUDAQ_SOURCE_DIR}/runtime
        )

#################################################################################

include_directories(BEFORE
        ${CMAKE_SOURCE_DIR}/include
)

add_mlir_library(MQSSPasses SHARED
    src/CustomExamplePass.cpp
    src/PrintQuakeGatesPass.cpp
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransforms
    MLIRSupport)

#add tests
add_subdirectory(tests)
