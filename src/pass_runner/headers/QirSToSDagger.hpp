/**
 * @file QirSToSDagger.hpp
 * @brief Declaration of the 'QirSToSDagger' class.
 */

#pragma once

#include "PassModule.hpp"

namespace llvm {

/**
 * @class QirSToSDagger
 * @brief This pass replaces S found adjecent with Pauli gate with S dagger
 * gate. If S is adjecent with Z gate, Z is reduced.
 */
class QirSToSDaggerPass : public PassModule {
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
