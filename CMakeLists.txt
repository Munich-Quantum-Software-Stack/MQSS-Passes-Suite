cmake_minimum_required(VERSION 3.22.1)
project(
  QIRPasses
  VERSION 0.1
  DESCRIPTION "QIR Passes")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w -fPIC")

# Set the installation directory for the daemon
set(CMAKE_INSTALL_PREFIX $ENV{HOME}/bin CACHE PATH "..." FORCE)
set(INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Find QDMI package
find_package(qdmi REQUIRED)

# Find QInfo package
find_package(qinfo REQUIRED)

# Find the Threads package
find_package(Threads REQUIRED)

# Find the QMAP package
find_package(mqt-qmap REQUIRED)

# Find the LLVM package
find_package(LLVM REQUIRED CONFIG)

# Set LLVM_ENABLE_THREADS to ON
set(LLVM_ENABLE_THREADS ON)
add_definitions(${LLVM_DEFINITIONS})


# Include all transformation passes here
set(TRANSFORMATION_PASSES_SOURCE_FILES
    # A
    #QirAnnotateUnsupportedGates.cpp
    # B
    QirBarrierBeforeFinalMeasurements.cpp
    # C
    #QirCNotToHCZHDecomposition.cpp
    QirCommuteCnotRx.cpp
    QirCommuteRxCnot.cpp
    QirCommuteCnotX.cpp
    QirCommuteXCnot.cpp
    QirCommuteCnotZ.cpp
    QirCommuteZCnot.cpp
    #QirCZToHCnotHDecomposition.cpp
    # D
    QirDeferMeasurement.cpp
    QirDivisionByZero.cpp
    QirDoubleCnotCancellation.cpp
    # F
    #QirFunctionAnnotator.cpp
    QirFunctionReplacement.cpp
    # G
    #QirGrouping.cpp
    # H
    QirHadamardAndXGateSwitch.cpp
    QirHadamardAndYGateSwitch.cpp
    QirHadamardAndZGateSwitch.cpp
    QirHXHToZ.cpp
    QirHZHToX.cpp
    # M
    QirMergeRotations.cpp
    # N
    QirNormalizeArgAngle.cpp
    QirNullRotationCancellation.cpp
    # P
    #QirPlaceIrreversibleGatesInMetadata.cpp
    QirPauliGateAndHadamardSwitch.cpp
    ## Q
    QirQMap.cpp
    QirQubitRemap.cpp
    # R
    #QirRedundantGatesCancellation.cpp
    QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
    #QirRemoveNonEntrypointFunctions.cpp
    QirReplaceConstantBranches.cpp
    QirResourceAnnotation.cpp
    QirReverseCnot.cpp
    #QirRzToRxRyRxDecomposition.cpp
    # S
    QirSDaggerToS.cpp
    QirSToSDagger.cpp
    QirSwapAndCnotReplacement.cpp
    #QirSwapToCnotsDecomposition.cpp
    # U
   #QirU3Decomposition.cpp
   #QirU3ToRzRyRzDecomposition.cpp
    # X
    QirXCnotXReduction.cpp
    QirXGateAndHadamardSwitch.cpp
    QirXYXDecomposition.cpp
    # Y
    QirYGateAndHadamardSwitch.cpp
    # Z
    QirZGateAndHadamardSwitch.cpp
    QirZXZDecomposition.cpp
    QirZYZDecomposition.cpp
)

# Include all analysis passes here
set(ANALYSIS_PASSES_SOURCE_FILES
    # A
    QirAllocationAnalysis.cpp)
    # F
#    QirFunctionValidationAnalysis.cpp)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/lib/passes")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_INSTALL_PREFIX}/include"
  ${MQT_QMAP_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIRS})

add_library(utilities SHARED src/utilities.cpp)

# Install the shared library
install(TARGETS utilities
        DESTINATION "${INSTALL_PATH}/lib/passes"
        )

 install(FILES include/PassModule.hpp DESTINATION include)

# Compile each pass as a shared object and link it
# to the required libraries
foreach(src_file ${ANALYSIS_PASSES_SOURCE_FILES})
  get_filename_component(src_name "src/${src_file}" NAME_WE)
  set(target_name "${src_name}Pass")
  add_library(${target_name} SHARED "src/${src_file}")

  # Link this pass with the required libraries
  target_link_libraries(${target_name} PRIVATE
    qdmi
    qinfo
    #fomac
    utilities)

  target_include_directories(${target_name}
    PRIVATE ${CMAKE_INSTALL_PREFIX}/include
            ${QDMI_INCLUDE_DIRS}
            ${QMAP_INCLUDE_DIRS}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/passes"
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  )
endforeach()

foreach(src_file ${TRANSFORMATION_PASSES_SOURCE_FILES})
  get_filename_component(src_name "src/${src_file}" NAME_WE)
  set(target_name "${src_name}Pass")
  add_library(${target_name} SHARED "src/${src_file}")

  set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Link this pass with the required libraries
  target_link_libraries(${target_name} PRIVATE
    qdmi
    qinfo
    #fomac
    utilities
  )

  #target_compile_features(${target_name} PUBLIC c_std_99)

  # Set include directories for the library
  target_include_directories(${target_name}
    PRIVATE ${CMAKE_INSTALL_PREFIX}/include
            ${QDMI_INCLUDE_DIRS}
            ${MQT_QMAP_INCLUDE_BUILD_DIR}
            ${QMAP_INCLUDE_DIRS}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/passes"
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  )

endforeach()

# Analysis passes invoked from custom passes have to be linked to the custom
# pass
target_link_libraries(QirQubitRemapPass PRIVATE QirAllocationAnalysisPass)
target_link_libraries(QirResourceAnnotationPass PRIVATE QirAllocationAnalysisPass)
target_link_libraries(QirZXZDecompositionPass PRIVATE QirZYZDecompositionPass)
target_link_libraries(QirXYXDecompositionPass PRIVATE QirZYZDecompositionPass)
#target_link_libraries(QirU3DecompositionPass PRIVATE QirZYZDecompositionPass)
target_link_libraries(QirQMapPass PRIVATE MQT::Core) 
#mqt-qmap-qdmi MQT::ProjectOptions MQT::ProjectWarnings)

#add_custom_target(
#  uninstall
#  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Build the documentation
option(BUILD_WITH_DOCS "Generate Docs" OFF)
if(BUILD_WITH_DOCS)
  set(BUILD_DOCS_STANDALONE OFF)
  message("Building documentation.")
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/docs)
endif()
