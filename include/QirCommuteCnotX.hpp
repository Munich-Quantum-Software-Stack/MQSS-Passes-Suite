/**
 * @file QirCommuteCnotX.hpp
 * @brief Declaration of the 'QirCommuteCnotXPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirCommuteCnotXPass
 * @brief This pass swaps adjacent CNOT and X gates whenever found in this
 * specific order.
 */
class QirCommuteCnotXPass : public PassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
