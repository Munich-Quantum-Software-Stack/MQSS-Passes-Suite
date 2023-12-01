# Include all transformation passes here
set(PASSES_SOURCE_FILES
    #[[
    # A
    QirAnnotateUnsupportedGates.cpp
    # B
    QirBarrierBeforeFinalMeasurements.cpp
    # C
    QirCNotToHCZHDecomposition.cpp
    QirCommuteCnotRx.cpp
    QirCommuteRxCnot.cpp
    QirCommuteCnotX.cpp
    QirCommuteXCnot.cpp
    QirCommuteCnotZ.cpp
    QirCommuteZCnot.cpp
    # D
    QirDeferMeasurement.cpp
    QirDivisionByZero.cpp
    QirDoubleCnotCancellation.cpp
    # F
    QirFunctionAnnotator.cpp
    QirFunctionReplacement.cpp
    # G
    QirGrouping.cpp
    # H
    QirHadamardAndXGateSwitch.cpp
    QirHadamardAndYGateSwitch.cpp
    QirHadamardAndZGateSwitch.cpp
    # M
    QirMergeRotations.cpp
    # N
    QirNormalizeArgAngle.cpp
    QirNullRotationCancellation.cpp
    # P
    QirPlaceIrreversibleGatesInMetadata.cpp
    # Q
    QirQubitRemap.cpp
    # R
    QirRedundantGatesCancellation.cpp
    QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
    QirRemoveNonEntrypointFunctions.cpp
    QirReplaceConstantBranches.cpp
    QirResourceAnnotation.cpp
    QirRzToRxRyRxDecomposition.cpp
    # S
    QirSDaggerToS.cpp
    QirSToSDagger.cpp
    # U
    QirU3ToRzRyRzDecomposition.cpp
    # X
    QirXCnotXReduction.cpp
    QirXGateAndHadamardSwitch.cpp
    # Y
    QirYGateAndHadamardSwitch.cpp
    # Z
    QirZGateAndHadamardSwitch.cpp
]]
    #
    QirU3Decomposition.cpp QirXYXDecomposition.cpp QirZXZDecomposition.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../headers)

# Include all analysis passes here

set(ANALYSIS_PASSES_SOURCE_FILES
    # A
    QirAllocationAnalysis.cpp
    # F
    QirFunctionValidation.cpp QirZYZDecomposition.cpp)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/bin/lib/pass_runner/passes")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(utilities SHARED utilities.cpp)

# Install the shared library
install(TARGETS utilities
        DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")

# Compile each pass as a shared object and link it to qdmi
foreach(src_file ${ANALYSIS_PASSES_SOURCE_FILES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(target_name "${src_name}Pass")

  add_library(${target_name} SHARED ${src_file})

  # Link the library with qdmi
  target_link_libraries(${target_name} PRIVATE qdmi utilities)

  # Set RPATH for this scheduler
  set_target_properties(
    ${target_name}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib/qdmi"
               INSTALL_RPATH_USE_LINK_PATH TRUE)

  # Set include directories for the library
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/qdmi)

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")

endforeach()
# Compile each pass as a shared object and link it to qdmi and all analysis
# passes
set(INSTALL_LIB_PATHS "${CMAKE_INSTALL_PREFIX}/bin/lib/pass_runner/passes;"
                      "${CMAKE_INSTALL_PREFIX}/bin/lib/qdmi")
string(REPLACE ";" "\\;" RLIBPATHS "${INSTALL_LIB_PATHS}")
foreach(src_file ${PASSES_SOURCE_FILES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(target_name "${src_name}Pass")

  add_library(${target_name} SHARED ${src_file})

  # Link the library with qdmi
  target_link_libraries(${target_name} PRIVATE qdmi utilities)

  # Set RPATH for this scheduler
  set_target_properties(
    ${target_name} PROPERTIES INSTALL_RPATH ${RLIBPATHS}
                              INSTALL_RPATH_USE_LINK_PATH TRUE)

  # Set include directories for the library
  target_include_directories(
    ${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/qdmi
                           ${CMAKE_SOURCE_DIR}/src/pass_runner/passes)

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")

endforeach()

# Analysis passes invoked from custom passes have to be linked to the custom
# pass target_link_libraries(QirQubitRemapPass PRIVATE
# QirAllocationAnalysisPass) target_link_libraries(QirResourceAnnotationPass
# PRIVATE QirAllocationAnalysisPass)
target_link_libraries(QirZXZDecompositionPass PRIVATE QirZYZDecompositionPass)
target_link_libraries(QirXYXDecompositionPass PRIVATE QirZYZDecompositionPass)
target_link_libraries(QirU3DecompositionPass PRIVATE QirZYZDecompositionPass)
