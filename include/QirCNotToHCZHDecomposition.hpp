/**
 * @file QirCNotToHCZHDecomposition.hpp
 * @brief Declaration of the 'QirCNotToHCZHDecompositionPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirCNotToHCZHDecompositionPass
 * @brief This pass decomposes a CNot gate into H, Cz, H gates.
 */
class QirCNotToHCZHDecompositionPass : public AgnosticPassModule
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
