/**
 * @file QirHadamardAndXGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndXGateSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirHadamardAndXGateSwitchPass
 * @brief This pass swaps adjacent H and X gates whenever found in this order.
 * As a result, X gate is changed into Z gate.
 */
class QirHadamardAndXGateSwitchPass : public PassModule {
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
