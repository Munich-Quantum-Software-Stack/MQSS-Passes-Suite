#include "QirPassManager.h"
#include <iostream>
#include <algorithm>

using namespace llvm;

QirPassManager::QirPassManager() {}

void QirPassManager::append(std::string pass) {
    passes_.push_back(pass);
}

PreservedAnalyses QirPassManager::run(Module *module, ModuleAnalysisManager &mam) {
    PreservedAnalyses allPassesPreserved;

    while (!passes_.empty()) {
        auto pass = passes_.back();
		void *soHandle = dlopen(pass.c_str(), RTLD_NOW /*RTLD_LAZY*/);

        std::cout << "Applying pass: " << pass << std::endl;

        if(!soHandle) {
            std::cout << "Warning: Error loading shared object: " << pass << std::endl;
            passes_.pop_back();
            continue;
        }

        using passCreator = PassModule* (*)();

        passCreator createQirPass = reinterpret_cast<passCreator>(dlsym(soHandle, "createQirPass"));

        if(!createQirPass) {
            std::cout << "Warning: Error getting factory function of pass: " << pass << std::endl;
            passes_.pop_back();
            dlclose(soHandle);
            continue;
        }

        PassModule *QirPass = createQirPass();

        QirPass->run(module, mam);
        
        delete QirPass;
        dlclose(soHandle);
        passes_.pop_back();
    }

    return allPassesPreserved;
}

