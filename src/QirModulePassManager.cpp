#include "QirModulePassManager.hpp"
#include <iostream>
#include <algorithm>

using namespace llvm;

QirModulePassManager::QirModulePassManager() {}

QirModulePassManager &QirModulePassManager::getInstance() {
    static QirModulePassManager instance;
    return instance;
}

void QirModulePassManager::setMetadata(const QirMetadata &metadata) {
    qirMetadata_ = metadata;
}

void QirModulePassManager::clearMetadata() {
    qirMetadata_.suitablePasses.clear();
    qirMetadata_.reversibleGates.clear();
}

QirMetadata &QirModulePassManager::getMetadata() {
    return qirMetadata_;
}

void QirModulePassManager::append(std::string pass) {
    passes_.push_back(pass);
}

void /*PreservedAnalyses*/ QirModulePassManager::run(Module &module, ModuleAnalysisManager &MAM) {
    //PreservedAnalyses allPassesPreserved; // TODO

    while (!passes_.empty()) {
        auto pass = passes_.back();
		void *soHandle = dlopen(pass.c_str(), /*RTLD_NOW*/ RTLD_LAZY);

        if(!soHandle) {
            std::cout << "Warning: Error loading shared object: " << pass << std::endl;
            passes_.pop_back();
            continue;
        }

        std::cout << "Applying pass: " << pass << std::endl;

        using passCreator = PassModule* (*)();

        passCreator createQirPass = reinterpret_cast<passCreator>(dlsym(soHandle, "createQirPass"));

        if(!createQirPass) {
            std::cout << "Warning: Error getting factory function of pass: " << pass << std::endl;
            passes_.pop_back();
            dlclose(soHandle);
            continue;
        }

        PassModule *QirPass = createQirPass();

        QirPass->run(module, MAM);
        
        delete QirPass;
        dlclose(soHandle);
        passes_.pop_back();
    }

    //return allPassesPreserved;
}

