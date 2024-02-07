/**
 * @file QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.hpp
 * @brief Declaration of the
 * 'QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass' class.
 */

#pragma once

#include <PassModule.hpp>

#include <algorithm>

namespace llvm
{

/**
 * @class QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass
 * @brief This pass removes all blocks with a single (terminator) instruction.
 */
class QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass
    : public PassModule
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
