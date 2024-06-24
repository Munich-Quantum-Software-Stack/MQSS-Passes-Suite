/**
 * @file QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp
 * @brief Implementation of the
 * 'QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes all blocks with a single (terminator) instruction.
 */

#include <QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass::run(
    Module &module, ModuleAnalysisManager &MAM)
{
    for (auto &function : module)
    {
        // Collect blocks that will be removed
        std::vector<BasicBlock *> useless_blocks;
        for (auto &block : function)
        {
            if (block.size() == 1)
            {
                auto *terminator = block.getTerminator();

                if (!terminator)
                    continue;

                if (auto *BI = dyn_cast<BranchInst>(terminator))
                    if (BI->isUnconditional())
                        useless_blocks.push_back(&block);
            }
        }

        // Remove useless blocks
        while (!useless_blocks.empty())
        {
            auto *useless_block = useless_blocks.back();
            auto *terminator = useless_block->getTerminator();
            if (auto *BI = dyn_cast<BranchInst>(terminator))
            {
                auto *useless_block_succesor = BI->getSuccessor(0);
                useless_block->replaceAllUsesWith(useless_block_succesor);
                useless_block->eraseFromParent();
            }
            useless_blocks.pop_back();
        }
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the
 * 'QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass' as a
 * 'PassModule'.
 * @return QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass();
}
