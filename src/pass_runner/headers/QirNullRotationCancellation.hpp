/**
 * @file QirNullRotationCancellation.hpp
 * @brief Declaration of the 'QirNullRotationCancellationPass' class.
 */

#pragma once

#include "PassModule.hpp"

#include <cmath>
#include <unordered_set>

namespace llvm
{

/**
 * @class QirNullRotationCancellationPass
 * @brief This pass removes rotation gates with null
 * rotation, that is rotation by 0 or by 2pi multiplies.
 */
class QirNullRotationCancellationPass : public PassModule
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
