#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Module.h>

#include "PassModule.h"

namespace llvm {

class QirRemoveNonEntrypointFunctionsPass : public PassModule {
public:
    PreservedAnalyses run(Module *module, ModuleAnalysisManager &/*mam*/);
};

}

