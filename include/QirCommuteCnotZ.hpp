/**
 * @file QirCommuteCnotZ.hpp
 * @brief Declaration of the 'QirCommuteCnotZPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirCommuteCnotZPass
 * @brief This pass swaps adjacent CNOT and Z gates whenever found in this
 * specific order.
 */
class QirCommuteCnotZPass : public AgnosticPassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

} // namespace llvm
