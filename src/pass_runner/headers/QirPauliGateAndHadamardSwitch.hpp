/**
 * @file QirPauliGateAndHadamardSwitch.cpp
 * @brief Declaration of the 'QirPauliGateAndHadamardSwitchPass' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirPauliGateAndHadamardSwitchPass
 * @brief This pass swaps adjacent Pauli gates and H whenever found in this
 * order. As a result, Pauli gate is changed accordingly.
 */
class QirPauliGateAndHadamardSwitchPass : public PassModule
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
