/**
 * @file selector_all.cpp
 * @brief Implementation of a dummy selector.
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief The main entry point of the program.
 *
 * The Selector.
 *
 * @return std::vector<std::string>
 */
extern "C" std::vector<std::string> selector(void) {
  // Append the desired passes
  std::vector<std::string> passes{
      "libQirDivisionByZeroPass.so",
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
      "libQirGroupingPass.so",
      "libQirRemoveNonEntrypointFunctionsPass.so",
      "libQirDeferMeasurementPass.so",
      "libQirBarrierBeforeFinalMeasurementsPass.so",
      "libQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass.so",
      "libQirQubitRemapPass.so",
      "libQirResourceAnnotationPass.so",
      "libQirNullRotationCancellationPass.so",
      "libQirMergeRotationsPass.so",
      "libQirDoubleCnotCancellationPass.so",
      "libQirHadamardAndXGateSwitchPass.so",
      "libQirHadamardAndYGateSwitchPass.so",
      "libQirHadamardAndZGateSwitchPass.so",
      "libQirXGateAndHadamardSwitchPass.so",
      "libQirYGateAndHadamardSwitchPass.so",
      "libQirZGateAndHadamardSwitchPass.so",
      "libQirSToSDaggerPass.so",
      "libQirSDaggerToSPass.so",
      "libQirZYZTransformPass.so",
  };

  std::cout
      << "[Selector].........Returning list of passes to the Selector Runner"
      << std::endl;

  std::reverse(passes.begin(), passes.end());
  return passes;
}
