#pragma once

#include "llvm.hpp"
#include <qdmi.hpp>
#include "../QirPassRunner.hpp"

using namespace llvm;

class PassModule {
public:
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM) = 0;
    virtual ~PassModule() {}
};

