/**
 * @file QirHadamardAndYGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndYGateSwitch' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirHadamardAndYGateSwitch
 * @brief This pass swaps adjacent H and Y gates whenever found in this order.
 */
class QirHadamardAndYGateSwitchPass : public AgnosticPassModule
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
