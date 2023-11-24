/**
 * @file QirReverseCnot.hpp
 * @brief Declaration of the 'QirReverseCnot' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirReverseCnot
 * @brief This pass replaces Cnot gate with a reversed Cnot surrounded by
 * Hadamard gates.
 */
class QirReverseCnotPass : public PassModule {
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
