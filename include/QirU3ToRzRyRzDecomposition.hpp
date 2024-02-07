/**
 * @file QirU3ToRzRyRzDecomposition.hpp
 * @brief Declaration of the 'QirU3ToRzRyRzDecompositionPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirU3ToRzRyRzDecompositionPass
 * @brief This pass decomposes a U3 gate into Rx and Ry gates.
 */
class QirU3ToRzRyRzDecompositionPass : public PassModule
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
