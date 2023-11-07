/**
 * @file selector_all.cpp
 * @brief Implementation of a dummy selector.
 */

#include <string>
#include <vector>

/**
 * @brief The main entry point of the program.
 *
 * The Selector.
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> selector(void) {
    // Append the desired passes
    std::vector<std::string> passes {
        "libQirNormalizeArgAnglePass.so",
        "libQirXCnotXReductionPass.so",
        "libQirCommuteCnotRxPass.so",
        "libQirCommuteRxCnotPass.so",
        "libQirCommuteCnotXPass.so",
        "libQirCommuteXCnotPass.so",
        "libQirCommuteCnotZPass.so",
        "libQirCommuteZCnotPass.so",
        "libQirPlaceIrreversibleGatesInMetadataPass.so",
	    "libQirAnnotateUnsupportedGatesPass.so",
        "libQirU3ToRzRyRzDecompositionPass.so",
        "libQirRzToRxRyRxDecompositionPass.so",
        "libQirCNotToHCZHDecompositionPass.so",
	    "libQirFunctionAnnotatorPass.so",
        "libQirRedundantGatesCancellationPass.so",
        "libQirFunctionReplacementPass.so",
        "libQirReplaceConstantBranchesPass.so",
        "libQirGroupingPass.so", // TODO: Does __quantum__rt__initialize belong to post-quantum?
	    "libQirRemoveNonEntrypointFunctionsPass.so",
        "libQirDeferMeasurementPass.so",
        "libQirBarrierBeforeFinalMeasurementsPass.so",
        "libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so",
        "libQirQubitRemapPass.so",
        "libQirResourceAnnotationPass.so",
	    "libQirNullRotationCancellationPass.so",
	    "libQirMergeRotationsPass.so",
        "libQirDoubleCnotCancellationPass.so",
    };

    return passes;
}

