/**
 * @file QirSwapToCnotsDecomposition.hpp
 * @brief Declaration of the 'QirSwapToCnotsDecompositionPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirSwapToCnotsDecompositionPass
 * @brief This pass decomposes a Swap gate into three Cnot gates.
 */
class QirSwapToCnotsDecompositionPass : public AgnosticPassModule
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
