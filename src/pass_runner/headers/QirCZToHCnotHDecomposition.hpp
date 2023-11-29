/**
 * @file QirCZToHCnotHDecomposition.hpp
 * @brief Declaration of the 'QirCZToHCnotHDecompositionPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirCZToHCnotHDecompositionPass
 * @brief This pass decomposes a CZ gate into H, Cnot, H gates.
 */
class QirCZToHCnotHDecompositionPass : public PassModule
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
