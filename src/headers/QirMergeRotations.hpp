#pragma once

#include "PassModule.hpp"

#include <unordered_set>

namespace llvm {

class QirMergeRotationsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

