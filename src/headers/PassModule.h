#pragma once

#include "llvm.hpp"

using namespace llvm;

class PassModule {
public:
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam) = 0;
    virtual ~PassModule() {}
};

