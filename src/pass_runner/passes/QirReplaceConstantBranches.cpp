/**
 * @file QirReplaceConstantBranches.cpp
 * @brief Implementation of the 'QirReplaceConstantBranchesPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirReplaceConstantBranches.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes those blocks with conditional
 * branching terminators with hard-coded conditions.
 */

#include "../headers/QirReplaceConstantBranches.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirReplaceConstantBranchesPass::run(Module &module,
                                    ModuleAnalysisManager & /*MAM*/) {
  // XXX THIS IS OUR CUSTOM PASS:

  for (auto &function : module) {
    std::vector<BasicBlock *> useless_blocks;
    for (auto &block : function) {
      if (auto *BI = dyn_cast<BranchInst>(block.getTerminator())) {
        if (BI->isConditional()) {
          if (auto *CI = dyn_cast<ConstantInt>(BI->getCondition())) {
            auto *trueTarget = BI->getSuccessor(0);
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

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);

  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel::O0);

  MPM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
  MPM.addPass(createModuleToFunctionPassAdaptor(JumpThreadingPass()));

  MPM.run(module, MAM);
  MPM = ModulePassManager();

  return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirReplaceConstantBranchesPass' as
 * a 'PassModule'.
 * @return QirReplaceConstantBranchesPass
 */
extern "C" PassModule *loadQirPass() {
  return new QirReplaceConstantBranchesPass();
}
