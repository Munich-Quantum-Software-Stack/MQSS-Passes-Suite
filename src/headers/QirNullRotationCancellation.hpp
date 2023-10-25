#pragma once

#include "PassModule.hpp"

#include <cmath>
#include <unordered_set>

namespace llvm {

class QirNullRotationCancellationPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

