#pragma once

#include "PassModule.h"

namespace llvm {

class QirCXCancellationPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &/*mam*/);
};

}

