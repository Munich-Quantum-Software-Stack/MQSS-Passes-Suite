/**
 * @file QirXCnotXReduction.hpp
 * @brief Declaration of the 'QirXCnotXReductionPass' class.
 */
#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirXCnotXReductionPass
 * @brief This pass removes X gates surrounding a CNOT gate.
 */
class QirXCnotXReductionPass : public PassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
