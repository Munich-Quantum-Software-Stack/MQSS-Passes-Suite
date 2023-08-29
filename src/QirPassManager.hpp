#ifndef QIR_PASS_MANAGER_H
#define QIR_PASS_MANAGER_H

#include "headers/PassModule.h"
#include <dlfcn.h>

using namespace llvm;

class QirPassManager {
public:
    QirPassManager();
    
    std::vector<std::string> getPasses() const {
        return passes_;
    }
    
    void append(std::string pass);
    PreservedAnalyses run(Module &module, ModuleAnalysisManager &mam);

private:
    std::vector<std::string> passes_;
};

#endif // QIR_PASS_MANAGER_H
