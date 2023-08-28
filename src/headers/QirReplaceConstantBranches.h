#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "PassModule.h"

namespace llvm {

class QirReplaceConstantBranchesPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam);
};

}
