/**
 * @file QirHadamardAndYGateSwitch.hpp
 * @brief Declaration of the 'QirHadamardAndYGateSwitch' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
<<<<<<<< HEAD:src/pass_runner/headers/QirPauliGateAndHadamardSwitch.hpp
 * @class QirPauliGateAndHadamardSwitch
 * @brief This pass swaps adjacent Pauli gates and H whenever found in this
 * order. As a result, Pauli gate is changed accordingly.
========
 * @class QirHadamardAndYGateSwitch
 * @brief This pass swaps adjacent H and Y gates whenever found in this order.
>>>>>>>> e4a2254 (Hadamard and Pauli gate switching split separately for each
gate):src/pass_runner/headers/QirHadamardAndYGateSwitch.hpp
 */
class QirHadamardAndYGateSwitchPass : public PassModule {
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
