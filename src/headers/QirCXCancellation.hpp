#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirCXCancellationPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

