/**
 * @file QirXGateAndHadamardSwitch.hpp
 * @brief Declaration of the 'QirXGateAndHadamardSwitch' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirXGateAndHadamardSwitch
 * @brief This pass swaps adjacent X gates and H whenever found in this order.
 * As a result, X gate is changed into Z gate.
 */
class QirXGateAndHadamardSwitchPass : public AgnosticPassModule
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
