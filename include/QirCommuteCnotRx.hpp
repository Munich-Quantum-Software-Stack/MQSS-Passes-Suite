/**
 * @file QirCommuteCnotRx.hpp
 * @brief Declaration of the 'QirCommuteCnotRxPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirCommuteCnotRxPass
 * @brief This pass swaps adjacent CNOT and Rx gates whenever found in this
 * specific order.
 */
class QirCommuteCnotRxPass : public AgnosticPassModule
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
