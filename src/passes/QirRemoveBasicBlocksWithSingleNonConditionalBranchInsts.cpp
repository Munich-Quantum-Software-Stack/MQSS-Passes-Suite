#include "../headers/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.hpp"

using namespace llvm;

PreservedAnalyses QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass::run(Module &module, ModuleAnalysisManager &MAM) {
    for (auto &function : module) {
        // Collect blocks that will be removed
        std::vector<BasicBlock*> useless_blocks;
        for (auto &block : function) {
            if (block.size() == 1) {
                auto *terminator = block.getTerminator();

                if (!terminator)
                    continue;                

                if (auto *BI = dyn_cast<BranchInst>(terminator))
                    if (BI->isUnconditional())
                        useless_blocks.push_back(&block);
            }
        }
        
		// Remove useless blocks
        while(!useless_blocks.empty()){
            auto *useless_block = useless_blocks.back();
			auto *terminator = useless_block->getTerminator();
			if (auto *BI = dyn_cast<BranchInst>(terminator)) {
				auto *useless_block_succesor = BI->getSuccessor(0);
				useless_block->replaceAllUsesWith(useless_block_succesor);
				useless_block->eraseFromParent();
			}
	        useless_blocks.pop_back();
        }
    }

    return PreservedAnalyses::none();
}

extern "C" PassModule* loadQirPass() {
    return new QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass();
}

