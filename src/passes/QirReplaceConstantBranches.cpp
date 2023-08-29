#include "../headers/QirReplaceConstantBranches.h"

using namespace llvm;

PreservedAnalyses QirReplaceConstantBranchesPass::run(Module &module, ModuleAnalysisManager &mam) {
    for (auto &function : module) {
        //std::vector<BasicBlock*> useless_blocks;
        for (auto &block : function) {
            if (auto *BI = dyn_cast<BranchInst>(block.getTerminator())) {
                if (BI->isConditional()) {
                    if (auto *CI = dyn_cast<ConstantInt>(BI->getCondition())) {
                        auto *trueTarget  = BI->getSuccessor(0);
                        auto *falseTarget = BI->getSuccessor(1);
                        if (CI->isOne()) {
                            ReplaceInstWithInst(BI, BranchInst::Create(trueTarget));
                            /*if (falseTarget->getSinglePredecessor())
                                useless_blocks.push_back(falseTarget);*/
                        } else {
                            ReplaceInstWithInst(BI, BranchInst::Create(falseTarget));
                            /*if (trueTarget->getSinglePredecessor())
                                useless_blocks.push_back(trueTarget);*/
                        }
                    }

                }
            }
        }
        
        /*while (!useless_blocks.empty()) {
            auto *useless_block = useless_blocks.back();
            errs() << "To be erased: " << useless_block->getName() << '\n';
            useless_block->eraseFromParent();
            useless_blocks.pop_back();
        }*/
    }
        
    ModulePassManager MPM;
    FunctionPassManager FPM;
    LoopPassManager LPM;
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;

    PB.registerModuleAnalyses(mam);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, mam);

    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(InstCombinePass());
    FPM.addPass(JumpThreadingPass());
    
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

    MPM.run(module, mam);

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirReplaceConstantBranchesPass();
}
