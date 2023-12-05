/**
 * @file QirU3Decomposition.hpp
 * @brief Declaration of the 'QirU3Decomposition' class.
 */
#pragma once

#include "PassModule.hpp"
#include "utilities.hpp"

namespace llvm
{

/**
 * @class QirXCnotXReductionPass
 * @brief This pass removes X gates surrounding a CNOT gate.
 */
class QirU3DecompositionPass : public PassModule
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
