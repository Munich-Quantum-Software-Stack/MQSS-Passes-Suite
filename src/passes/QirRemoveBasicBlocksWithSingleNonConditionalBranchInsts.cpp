#include "../headers/QirRemoveBasicBlocksWithSingleNonConditionalBranchInsts.h"

using namespace llvm;

PreservedAnalyses QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass::run(Module *module, ModuleAnalysisManager &/*mam*/){
    std::vector<BasicBlock*> useless_blocks;
    for(auto &function : *module){
        for(auto &block : function){
            if(block.size() == 1){
                auto *terminator = block.getTerminator();

                if(!terminator)
                    continue;                

                if(auto *BI = dyn_cast<BranchInst>(terminator))
                    if(BI->isUnconditional())
                        useless_blocks.push_back(&block);
            }
        }

        while(!useless_blocks.empty()){
            auto *useless_block            = useless_blocks.back();
            auto *terminator_useless_block = useless_block->getTerminator();

            if(!terminator_useless_block)
                continue;

            auto *BranchInst_useless_block = dyn_cast<BranchInst>(terminator_useless_block);
            auto *successor_useless_block  = BranchInst_useless_block->getSuccessor(0);

            for(auto *predecessor : predecessors(useless_block)){
                auto *terminator_predecessor = predecessor->getTerminator();

                if(auto *BranchInst_predecessor = dyn_cast<BranchInst>(terminator_predecessor))
                    for(unsigned i = 0; i < BranchInst_predecessor->getNumSuccessors(); ++i)
                        if(BranchInst_predecessor->getSuccessor(i) == useless_block)
                            BranchInst_predecessor->setSuccessor(i, successor_useless_block);
            }

            useless_block->eraseFromParent();
	        useless_blocks.pop_back();
        }
    }
    return PreservedAnalyses::all();
}

extern "C" PassModule* createQirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass() {
    return new QirRemoveBasicBlocksWithSingleNonConditionalBranchInstsPass();
}
