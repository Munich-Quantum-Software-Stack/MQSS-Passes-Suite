#include "../headers/QirReplaceConstantBranches.hpp"

using namespace llvm;

PreservedAnalyses QirReplaceConstantBranchesPass::run(Module &module, ModuleAnalysisManager &/*MAM*/) {
    // XXX THIS IS OUR CUSTOM PASS:

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

    // XXX THIS IS HOW YOU INVOKE LLVM PASSES FROM WITHIN OUR CUSTOM PASS:

    PassBuilder PB;

    LoopAnalysisManager     LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager    CGAM;
    ModuleAnalysisManager   MAM;

    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);

    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O0);

    MPM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
    MPM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
    MPM.addPass(createModuleToFunctionPassAdaptor(JumpThreadingPass()));
    
    MPM.run(module, MAM);
    MPM = ModulePassManager();
    
    // XXX THIS IS HOW YOU FETCH METADATA FROM THE MODULE:

    /*Metadata* metadataVersion = module.getModuleFlag("qir_major_version");
    if (metadataVersion)
        if (ConstantAsMetadata* intMetadata = dyn_cast<ConstantAsMetadata>(metadataVersion))
            if (ConstantInt* intConstant = dyn_cast<ConstantInt>(intMetadata->getValue()))
                errs() << "qir_major_version: " << intConstant->getSExtValue() << '\n';

    Metadata* metadataSupport = module.getModuleFlag("lrz_supports_qir");
    if (metadataSupport)
        if (ConstantAsMetadata* boolMetadata = dyn_cast<ConstantAsMetadata>(metadataSupport))
            if (ConstantInt* boolConstant = dyn_cast<ConstantInt>(boolMetadata->getValue()))
                errs() << "lrz_supports_qir: " << (boolConstant->isOne() ? "true" : "false") << '\n';*/

    // XXX THIS IS HOW YOU FETCH OUR OWN METADATA:

    // TODO

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirReplaceConstantBranchesPass();
}

