#include "../headers/QirReplaceConstantBranches.hpp"

using namespace llvm;

PreservedAnalyses QirReplaceConstantBranchesPass::run(Module &module, ModuleAnalysisManager &MAM) {
    for (auto &function : module) {
        std::vector<BasicBlock*> useless_blocks;
        for (auto &block : function) {
            if (auto *BI = dyn_cast<BranchInst>(block.getTerminator())) {
                if (BI->isConditional()) {
                    if (auto *CI = dyn_cast<ConstantInt>(BI->getCondition())) {
                        auto *trueTarget  = BI->getSuccessor(0);
                        auto *falseTarget = BI->getSuccessor(1);
                        if (CI->isOne()) {
                            if (falseTarget->hasNPredecessors(1))
                                useless_blocks.push_back(falseTarget);
                            ReplaceInstWithInst(BI, BranchInst::Create(trueTarget));
                        } else {
                            if (trueTarget->hasNPredecessors(1))
                                useless_blocks.push_back(trueTarget);
                            ReplaceInstWithInst(BI, BranchInst::Create(falseTarget));
                        }
                    }
                }
            }
        }
        
        while (!useless_blocks.empty()) {
            auto *useless_block = useless_blocks.back();
            useless_block->eraseFromParent();
            useless_blocks.pop_back();
        }
    }
        
    //ModulePassManager       MPM;
    //FunctionPassManager     FPM;
    //LoopPassManager         LPM;

    //ModuleAnalysisManager   MAM;
    //FunctionAnalysisManager FAM;
    //LoopAnalysisManager     LAM;
    //CGSCCAnalysisManager    CGAM;

    //PassBuilder PB;

    //PB.registerModuleAnalyses(MAM);
    //PB.registerFunctionAnalyses(FAM);
    //PB.registerLoopAnalyses(LAM);
    //PB.registerCGSCCAnalyses(CGAM);

    //PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    //FPM.addPass(SimplifyCFGPass());
    //FPM.addPass(InstCombinePass());
    //FPM.addPass(JumpThreadingPass());
    
    //MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

    //MPM.run(module, MAM);

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirReplaceConstantBranchesPass();
}
