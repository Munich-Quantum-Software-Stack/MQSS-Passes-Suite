#include "QirPassManager.h"

using namespace llvm;

QirPassManager::QirPassManager() {}

void QirPassManager::append(PassModule* pass) {
    passes_.push_back(pass);
}

PreservedAnalyses QirPassManager::run(Module *module, ModuleAnalysisManager &mam) {
    PreservedAnalyses allPassesPreserved;

    for(PassModule* pass : passes_) {
        PreservedAnalyses passPreserved = pass->run(module, mam);
        allPassesPreserved.intersect(passPreserved);
    }

    return allPassesPreserved;
}

