#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Module.h>

namespace llvm {

class PassModule {
public:
    virtual PreservedAnalyses run(Module *module, ModuleAnalysisManager &mam) = 0;
    virtual ~PassModule() {}
};

}

