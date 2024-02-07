/**
 * @file QirHadamardAndZGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndZGateSwitch' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirHadamardAndZGateSwitch
 * @brief This pass swaps adjacent H and Z gates whenever found in this order.
 * As a result, Z gate is changed into X gate.
 */
class QirHadamardAndZGateSwitchPass : public PassModule
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
