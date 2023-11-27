/**
 * @file QirHadamardAndXGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndXGateSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirHadamardAndPauliGateSwitchPass
 * @brief This pass swaps adjacent H and Pauli gates whenever found in this
 * order. As a result, Pauli gate is changed accordingly.
 */
class QirHadamardAndPauliGateSwitchPass : public PassModule {
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
