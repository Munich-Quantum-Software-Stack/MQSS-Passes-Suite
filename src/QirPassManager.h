#ifndef QIR_PASS_MANAGER_H
#define QIR_PASS_MANAGER_H

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Module.h>

#include "../src/headers/PassModule.h"

namespace llvm {

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

}

#endif // QIR_PASS_MANAGER_H

