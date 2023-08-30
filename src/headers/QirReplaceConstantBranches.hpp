#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirReplaceConstantBranchesPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

