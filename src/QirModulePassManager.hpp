#ifndef QIR_MODULE_PASS_MANAGER_H
#define QIR_MODULE_PASS_MANAGER_H

#include "headers/PassModule.hpp"

#include <dlfcn.h>

using namespace llvm;

class QirModulePassManager {
public:
    QirModulePassManager();
    
    std::vector<std::string> getPasses() const {
        return passes_;
    }
    
    void append(std::string pass);
    void /*PreservedAnalyses*/ run(Module &module, ModuleAnalysisManager &MAM);

private:
    std::vector<std::string> passes_;
};

#endif // QIR_MODULE_PASS_MANAGER_H
