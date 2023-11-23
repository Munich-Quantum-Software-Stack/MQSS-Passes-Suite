# Include all transformation passes here
set(PASSES_SOURCE_FILES
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
    QirZGateAndHadamardSwitch.cpp)

# Include all analysis passes here
set(ANALYSIS_PASSES_SOURCE_FILES
    # A
    QirAllocationAnalysis.cpp
    # F
    QirFunctionValidation.cpp)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/bin/lib/pass_runner/passes")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Compile each pass as a shared object and link it to FoMaC
foreach(src_file ${ANALYSIS_PASSES_SOURCE_FILES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(target_name "${src_name}Pass")

  add_library(${target_name} SHARED ${src_file})

  # Link the library with FoMaC
  target_link_libraries(${target_name} PRIVATE FoMaC qdmi)

  # Set RPATH for this pass
  set_target_properties(
    ${target_name}
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib/fomac"
               INSTALL_RPATH_USE_LINK_PATH TRUE)

  # Set include directories for the library
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/qdmi
                                                    ${CUSTOM_FOMAC_PATH})

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")

endforeach()

# Compile each pass as a shared object and link it to FoMaC and all analysis
# passes
set(INSTALL_LIB_PATHS "${CMAKE_INSTALL_PREFIX}/bin/lib/pass_runner/passes;"
                      "${CMAKE_INSTALL_PREFIX}/bin/lib/fomac")
string(REPLACE ";" "\\;" RLIBPATHS "${INSTALL_LIB_PATHS}")
foreach(src_file ${PASSES_SOURCE_FILES})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(target_name "${src_name}Pass")
  add_library(${target_name} MODULE ${src_file})

  # Link the library with FoMaC and QDMI
  target_link_libraries(${target_name} PRIVATE FoMaC qdmi)

  # Set RPATH for this pass
  set_target_properties(
    ${target_name} PROPERTIES INSTALL_RPATH ${RLIBPATHS}
                              INSTALL_RPATH_USE_LINK_PATH TRUE)

  # Set include directories for the library
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/qdmi
                                                    ${CUSTOM_FOMAC_PATH})

  # Install the shared library
  install(TARGETS ${target_name}
          DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")

endforeach()

# Analysis passes invoked from custom passes have to be linked to the custom
# pass
target_link_libraries(QirQubitRemapPass PRIVATE QirAllocationAnalysisPass)

target_link_libraries(QirResourceAnnotationPass
                      PRIVATE QirAllocationAnalysisPass)
