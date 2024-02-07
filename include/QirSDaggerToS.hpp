/**
 * @file QirSDaggerToS.hpp
 * @brief Declaration of the 'QirSDaggerToS' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirSDaggerToS
 * @brief This pass replaces S dagger found adjecent with Pauli gate with S
 * gate. If S dagger is adjecent with Z gate, Z is reduced.
 */
class QirSDaggerToSPass : public PassModule
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
