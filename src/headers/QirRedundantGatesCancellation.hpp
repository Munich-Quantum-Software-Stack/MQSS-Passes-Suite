#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirRedundantGatesCancellationPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

