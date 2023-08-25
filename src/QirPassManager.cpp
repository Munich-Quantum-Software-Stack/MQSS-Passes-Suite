#include "QirPassManager.h"
#include <iostream>

using namespace llvm;

QirPassManager::QirPassManager() {}

void QirPassManager::append(std::string pass) {
    passes_.push_back(pass);
}

PreservedAnalyses QirPassManager::run(Module *module, ModuleAnalysisManager &mam) {
    PreservedAnalyses allPassesPreserved;

    for(std::string pass : passes_) {
		void *soHandle = dlopen(pass.c_str(), RTLD_NOW /*RTLD_LAZY*/);

        if(!soHandle) {
            std::cout << "Warning: Error loading shared object: " << pass << std::endl;
            continue;
        }

        using passCreator = PassModule* (*)();

        passCreator createQirPass = reinterpret_cast<passCreator>(dlsym(soHandle, "createQirPass"));

        if(!createQirPass) {
            std::cout << "Warning: Error getting factory function of pass: " << pass << std::endl;
            dlclose(soHandle);
            continue;
        }

        PassModule *QirPass = createQirPass();

        QirPass->run(module, mam);
        
        delete QirPass;
        dlclose(soHandle);
    }

    return allPassesPreserved;
}

