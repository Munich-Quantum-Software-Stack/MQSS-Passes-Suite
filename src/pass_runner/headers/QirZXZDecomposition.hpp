/**
 * @file QirZXZDecomposition.hpp
 * @brief Declaration of the 'QirZXZDecomposition' class.
 */
#pragma once

#include "PassModule.hpp"
#include "utilities.hpp"

namespace llvm
{

/**
 * @class QirZXZDecompositionPass
 * @brief ZXZ Decomposition
 */
class QirZXZDecompositionPass : public PassModule
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
