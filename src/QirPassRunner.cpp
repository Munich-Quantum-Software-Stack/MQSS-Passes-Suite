#include "QirPassRunner.hpp"
#include <iostream>
#include <algorithm>

using namespace llvm;

QirPassRunner::QirPassRunner() {}

QirPassRunner &QirPassRunner::getInstance() {
    static QirPassRunner instance;
    return instance;
}

void QirPassRunner::setMetadata(const QirMetadata &metadata) {
    qirMetadata_ = metadata;
}

void QirPassRunner::clearMetadata() {
    qirMetadata_.suitablePasses.clear();
    qirMetadata_.reversibleGates.clear();
    qirMetadata_.supportedGates.clear();
    qirMetadata_.availablePlatforms.clear();
    qirMetadata_.injectedAnnotations.clear();
}

QirMetadata &QirPassRunner::getMetadata() {
    return qirMetadata_;
}

void QirPassRunner::append(std::string pass) {
    passes_.push_back(pass);
}

void /*PreservedAnalyses*/ QirPassRunner::run(Module &module, ModuleAnalysisManager &MAM) {
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

