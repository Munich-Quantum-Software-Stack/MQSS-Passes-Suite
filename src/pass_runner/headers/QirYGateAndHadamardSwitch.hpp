/**
 * @file QirYGateAndHadamardSwitch.hpp
 * @brief Declaration of the 'QirYGateAndHadamardSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirYGateAndHadamardSwitch
 * @brief This pass swaps adjacent Y gates and H whenever found in this order.
 */
class QirYGateAndHadamardSwitchPass : public PassModule
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
