//#pragma once

#ifndef PASS_MODULE_H
#define PASS_MODULE_H

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Module.h>

namespace llvm {

class PassModule {
public:
    virtual PreservedAnalyses run(Module *module, ModuleAnalysisManager &mam) = 0;
    virtual ~PassModule() {}
};

}

#endif // PASS_MODULE_H

