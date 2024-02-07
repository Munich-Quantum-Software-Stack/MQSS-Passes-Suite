/**
 * @file QirRzToRxRyRxDecomposition.hpp
 * @brief Declaration of the 'QirRzToRxRyRxDecompositionPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirRzToRxRyRxDecompositionPass
 * @brief This pass decomposes an Rz gate into Rx and Ry gates.
 */
class QirRzToRxRyRxDecompositionPass : public PassModule
{
  public:
    /**
     * @brief Applies this pass to the QIR's LLVM module.
     *
     * @param module The module of the submitted QIR.
     * @param MAM The module analysis manager.
     * @return PreservedAnalyses
     */
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM,
                          QDMI_Device dev);
};

} // namespace llvm
