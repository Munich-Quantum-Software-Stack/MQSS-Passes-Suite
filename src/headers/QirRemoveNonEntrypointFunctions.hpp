#pragma once

#include "PassModule.hpp"

namespace llvm {

class QirRemoveNonEntrypointFunctionsPass : public PassModule {
public:
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM);
};

}

