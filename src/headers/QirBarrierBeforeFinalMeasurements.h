#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "PassModule.h"

namespace llvm {

class QirBarrierBeforeFinalMeasurementsPass : public PassModule {
public:
    PreservedAnalyses run(Module *module, ModuleAnalysisManager &/*mam*/);
};

}

