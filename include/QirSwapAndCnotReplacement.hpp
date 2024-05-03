/**
 * @file QirSwapAndCnotReplacement.hpp
 * @brief Declaration of the 'QirSwapAndCnotReplacementPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirSwapAndCnotReplacementPass
 * @brief This pass replaces sequential Swap gate and Cnot gate acting on the
 * same qubits with the exact same Cnot followed by reversed Cnot.
 */
class QirSwapAndCnotReplacementPass : public AgnosticPassModule
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
