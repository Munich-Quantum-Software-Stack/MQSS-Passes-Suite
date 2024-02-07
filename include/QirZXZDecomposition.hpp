/**
 * @file QirZXZDecomposition.hpp
 * @brief Declaration of the 'QirZXZDecomposition' class.
 */
#pragma once

#include "utilities.hpp"
#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirZXZDecompositionPass
 * @brief This Pass creates a ZXZ Decomposition of given gates
 * (gatesToDecompose).
 */
class QirZXZDecompositionPass : public PassModule
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
