/**
 * @file QirHadamardAndXGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndXGateSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
<<<<<<<< HEAD:src/pass_runner/headers/QirHadamardAndPauliGateSwitch.hpp
 * @class QirHadamardAndPauliGateSwitch
 * @brief This pass swaps adjacent H and Pauli gates whenever found in this
 * order. As a result, Pauli gate is changed accordingly.
========
 * @class QirHadamardAndXGateSwitch
 * @brief This pass swaps adjacent H and X gates whenever found in this order.
As a result, X gate is changed into Z gate.
>>>>>>>> e4a2254 (Hadamard and Pauli gate switching split separately for each
gate):src/pass_runner/headers/QirHadamardAndXGateSwitch.hpp
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
