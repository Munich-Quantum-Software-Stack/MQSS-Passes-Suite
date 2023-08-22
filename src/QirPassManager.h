#ifndef QIR_PASS_MANAGER_H
#define QIR_PASS_MANAGER_H

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>

#include <dlfcn.h>

#include "../src/headers/PassModule.h"

using namespace llvm; //{

class QirPassManager {
public:
    QirPassManager();
    
    std::vector<PassModule*> getPasses() const {
        return passes_;
    }
    
    void append(PassModule* pass);
    PreservedAnalyses run(Module *module, ModuleAnalysisManager &mam);

private:
    std::vector<PassModule*> passes_;
};

//}

#endif // QIR_PASS_MANAGER_H

