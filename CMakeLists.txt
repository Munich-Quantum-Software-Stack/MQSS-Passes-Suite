cmake_minimum_required(VERSION 3.22.1)
project(
  QIRPasses
  VERSION 0.1
  DESCRIPTION "QIR Passes")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Set the installation directory for the daemon
#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin CACHE PATH "..." FORCE)
#endif()
set(INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(MQT_QMAP_QDMI_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Find QRM package
find_package(qrm REQUIRED)

# Find FoMaC package
find_package(fomac REQUIRED)

# Find QDMI package
find_package(qdmi REQUIRED)

# Find QInfo package
find_package(qinfo REQUIRED)

# Find the Threads package
find_package(Threads REQUIRED)

#Find the Python Package
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
if(NOT Python_FOUND)
  message(
    FATAL_ERROR "Python Package is not found. Please install using "
  )
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

# Add QMAP
add_subdirectory(extern/mqt-qmap)
add_subdirectory(src)

# Include all transformation passes here
set(TRANSFORMATION_PASSES_SOURCE_FILES
    ## A
    #QirAnnotateUnsupportedGates.cpp
    ## B
    #QirBarrierBeforeFinalMeasurements.cpp
    ## C
    #QirCNotToHCZHDecomposition.cpp
    #QirCommuteCnotRx.cpp
    #QirCommuteRxCnot.cpp
    #QirCommuteCnotX.cpp
    #QirCommuteXCnot.cpp
    #QirCommuteCnotZ.cpp
    #QirCommuteZCnot.cpp
    #QirCZToHCnotHDecomposition.cpp
    ## D
    #QirDeferMeasurement.cpp
    #QirDivisionByZero.cpp
   ##QirDoubleCnotCancellation.cpp
    ## F
    #QirFunctionAnnotator.cpp
    #QirFunctionReplacement.cpp
    ## G
    #QirGrouping.cpp
    ## H
    #QirHadamardAndXGateSwitch.cpp
    #QirHadamardAndYGateSwitch.cpp
    #QirHadamardAndZGateSwitch.cpp
    ## M
    #QirMergeRotations.cpp
    ## N
    #QirNormalizeArgAngle.cpp
    #QirNullRotationCancellation.cpp
    ## P
    #QirPlaceIrreversibleGatesInMetadata.cpp
    ## Q
    QirQmap.cpp
    #QirQubitRemap.cpp
    ## R
    #QirRedundantGatesCancellation.cpp
    #QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
    #QirRemoveNonEntrypointFunctions.cpp
    #QirReplaceConstantBranches.cpp
    #QirResourceAnnotation.cpp
    #QirReverseCnot.cpp
    #QirRzToRxRyRxDecomposition.cpp
    ## S
    #QirSDaggerToS.cpp
    #QirSToSDagger.cpp
    #QirSwapAndCnotReplacement.cpp
    #QirSwapToCnotsDecomposition.cpp
    ## U
   ##QirU3Decomposition.cpp
   ##QirU3ToRzRyRzDecomposition.cpp
    ## X
    #QirXCnotXReduction.cpp
    #QirXGateAndHadamardSwitch.cpp
    #QirXYXDecomposition.cpp
    ## Y
    #QirYGateAndHadamardSwitch.cpp
    ## Z
    #QirZGateAndHadamardSwitch.cpp
    #QirZXZDecomposition.cpp
    #QirZYZDecomposition.cpp
)

## Include all analysis passes here
#set(ANALYSIS_PASSES_SOURCE_FILES
#    # A
#    QirAllocationAnalysis.cpp
#    # F
#    QirFunctionValidationAnalysis.cpp)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/lib/passes")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_INSTALL_PREFIX}/include"
  ${LLVM_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  )

#add_library(utilities SHARED src/utilities.cpp)
#
## Install the shared library
#install(TARGETS utilities
#        DESTINATION "${INSTALL_PATH}/lib/passes")

## Compile each pass as a shared object and link it
## to the required libraries
#foreach(src_file ${ANALYSIS_PASSES_SOURCE_FILES})
#  get_filename_component(src_name "src/${src_file}" NAME_WE)
#  set(target_name "${src_name}Pass")
#  add_library(${target_name} SHARED "src/${src_file}")
#
#  # Link this pass with the required libraries
#  target_link_libraries(${target_name} PRIVATE
#    qdmi
#    qinfo
#    fomac
#    utilities)
#
#  target_include_directories(${target_name}
#    PRIVATE ${CMAKE_INSTALL_PREFIX}/include
#            ${CMAKE_BINARY_DIR}/_deps/qdmi-src/src #TODO
#    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#           $<INSTALL_INTERFACE:include>)
#
#  # Install the shared library
#  install(TARGETS ${target_name}
#          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/passes"
#          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
#  )
#endforeach()

foreach(src_file ${TRANSFORMATION_PASSES_SOURCE_FILES})
  get_filename_component(src_name "src/${src_file}" NAME_WE)
  set(target_name "${src_name}Pass")
  add_library(${target_name} SHARED "src/${src_file}")

  set_target_properties(QirQMapPass PROPERTIES POSITION_INDEPENDENT_CODE ON)
  #set_property(TARGET QirQMapPass PROPERTY POSITION_INDEPENDENT_CODE ON)

  # Link this pass with the required libraries
  target_link_libraries(${target_name} PRIVATE
    qdmi
    qinfo
    #fomac
    #utilities
    ${Python_LIBRARIES})
  )

  #target_compile_features(${target_name} PUBLIC c_std_99)

  # Set include directories for the library
  target_include_directories(${target_name}
    PRIVATE ${CMAKE_INSTALL_PREFIX}/include
            ${CMAKE_BINARY_DIR}/_deps/qdmi-src/src #TODO
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include> ${Python_INCLUDE_DIRS})

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/passes"
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  )

endforeach()

## Analysis passes invoked from custom passes have to be linked to the custom
## pass
#target_link_libraries(QirQubitRemapPass PRIVATE QirAllocationAnalysisPass)
#target_link_libraries(QirResourceAnnotationPass PRIVATE QirAllocationAnalysisPass)
#target_link_libraries(QirZXZDecompositionPass PRIVATE QirZYZDecompositionPass)
#target_link_libraries(QirXYXDecompositionPass PRIVATE QirZYZDecompositionPass)
##target_link_libraries(QirU3DecompositionPass PRIVATE QirZYZDecompositionPass)
target_link_libraries(QirQMapPass PRIVATE mqt-qmap-qdmi MQT::ProjectOptions MQT::ProjectWarnings)

#add_custom_target(
#  uninstall
#  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

## Build the documentation
#option(BUILD_WITH_DOCS "Generate Docs" OFF)
#if(BUILD_WITH_DOCS)
#  set(BUILD_DOCS_STANDALONE OFF)
#  message("Building documentation.")
#  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/docs)
#endif()
