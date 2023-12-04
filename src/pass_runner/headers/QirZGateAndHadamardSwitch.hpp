/**
 * @file QirZGateAndHadamardSwitch.hpp
 * @brief Declaration of the 'QirZGateAndHadamardSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm
{

/**
 * @class QirZGateAndHadamardSwitch
 * @brief This pass swaps adjacent Z gates and H whenever found in this order.
 * As a result, Z gate is changed into X gate.
 */
class QirZGateAndHadamardSwitchPass : public PassModule
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
