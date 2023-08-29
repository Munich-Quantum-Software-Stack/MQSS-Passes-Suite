#pragma once

#include "PassModule.h"

namespace llvm {

class QirReplaceConstantBranchesPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam);
};

}

