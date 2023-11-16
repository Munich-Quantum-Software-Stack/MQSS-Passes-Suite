# Include all new passes here
set(PASSES_SOURCE_FILES
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
    ## D
    #QirDeferMeasurement.cpp
    #QirDivisionByZero.cpp
    #QirDoubleCnotCancellation.cpp
    ## F
    #QirFunctionAnnotator.cpp
    #QirFunctionReplacement.cpp
    #QirFunctionValidation.cpp
    ## G
    #QirGrouping.cpp
    ## M
    #QirMergeRotations.cpp
    ## N
    #QirNormalizeArgAngle.cpp
    #QirNullRotationCancellation.cpp
    ## P
    #QirPlaceIrreversibleGatesInMetadata.cpp
    ## Q
    #QirQubitRemap.cpp
    ## R
    #QirRedundantGatesCancellation.cpp
    #QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
    #QirRemoveNonEntrypointFunctions.cpp
    #QirReplaceConstantBranches.cpp
    #QirResourceAnnotation.cpp
    #QirRzToRxRyRxDecomposition.cpp
    ## U
    #QirU3ToRzRyRzDecomposition.cpp
    ## X
    QirXCnotXReduction.cpp
)

# Set RPATH for the shared libraries in this directory
set(CMAKE_INSTALL_RPATH "${INSTALL_PATH}/bin/lib/pass_runner/passes")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Compile each pass as a shared object and link it to qdmi 
foreach(src_file ${PASSES_SOURCE_FILES})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(target_name "${src_name}Pass")
    add_library(${target_name} MODULE ${src_file})
    target_link_libraries(${target_name} PRIVATE qdmi)
endforeach()

## Analysis passes invoked from custom passes have to be linked to the custom pass
#add_library(QirAllocationAnalysisPass SHARED QirAllocationAnalysis.cpp)
#target_link_libraries(QirFunctionValidationPass PRIVATE QirAllocationAnalysisPass)
#target_link_libraries(QirQubitRemapPass PRIVATE QirAllocationAnalysisPass)
#target_link_libraries(QirResourceAnnotationPass PRIVATE QirAllocationAnalysisPass)
#
## Specify the installation directory for the shared libraries
#list(APPEND PASSES_SOURCE_FILES QirAllocationAnalysis.cpp)
foreach(src_file ${PASSES_SOURCE_FILES})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(target_name "${src_name}Pass")
    install(TARGETS ${target_name} DESTINATION "${INSTALL_PATH}/bin/lib/pass_runner/passes")
endforeach()

