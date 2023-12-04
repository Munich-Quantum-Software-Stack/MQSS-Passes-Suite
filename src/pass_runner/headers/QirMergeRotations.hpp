/**
 * @file QirMergeRotations.hpp
 * @brief Declaration of the 'QirMergeRotationsPass' class.
 */

#pragma once

#include "PassModule.hpp"

#include <unordered_set>

namespace llvm
{

/**
 * @class QirNullRotationCancellationPass
 * @brief This pass merges equivalent rotation gates into single
 * rotation.
 */
class QirMergeRotationsPass : public PassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

} // namespace llvm
