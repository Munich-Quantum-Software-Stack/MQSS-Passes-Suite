#pragma once

#include "llvm.hpp"

using namespace llvm;

class PassModule {
public:
    virtual PreservedAnalyses run(Module &module, ModuleAnalysisManager &MAM) = 0;
    virtual ~PassModule() {}
};

/*class AnalysisFunction {
public:
    virtual Result run(Function &function, FunctionAnalysisManager &FAM) = 0;
    virtual ~AnalysisFunction() {}
};*/

