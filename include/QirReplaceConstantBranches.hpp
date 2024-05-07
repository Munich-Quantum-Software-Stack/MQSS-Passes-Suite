/**
 * @file QirReplaceConstantBranches.hpp
 * @brief Declaration of the 'QirReplaceConstantBranchesPass' class.
 */

#pragma once

#include <PassModule.hpp>

namespace llvm
{

/**
 * @class QirReplaceConstantBranchesPass
 * @brief This pass removes those blocks with conditional branching
 * terminators with hard-coded conditions.
 */
class QirReplaceConstantBranchesPass : public AgnosticPassModule
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
