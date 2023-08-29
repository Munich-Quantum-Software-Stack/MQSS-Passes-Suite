#pragma once

#include "PassModule.h"

namespace llvm {

class QirRemoveNonEntrypointFunctionsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &/*mam*/);
};

}

