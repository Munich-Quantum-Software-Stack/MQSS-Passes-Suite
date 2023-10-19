#include "QirPassRunner.hpp"
#include <iostream>
#include <algorithm>
#include <string>

using namespace llvm;

// Default constructor of the 'QirPassRunner' class
QirPassRunner::QirPassRunner() {}

// Returns a reference to a 'QirPassRuner' object
QirPassRunner &QirPassRunner::getInstance() {
    static QirPassRunner instance;
    return instance;
}

/* Saves 'metadata' as the private metadata of the 'QirPassRunner' 
 * class
 */
void QirPassRunner::setMetadata(const QirMetadata &metadata) {
    qirMetadata_ = metadata;
}

// Empties all structures within the metadata
void QirPassRunner::clearMetadata() {
    qirMetadata_.reversibleGates.clear();
    qirMetadata_.supportedGates.clear();
    qirMetadata_.availablePlatforms.clear();
    qirMetadata_.injectedAnnotations.clear();
}

// Returns the private metadata of the 'QirPassRunner' class
QirMetadata &QirPassRunner::getMetadata() {
    return qirMetadata_;
}

/* Inserts a pass in the private vector 'passes_' of the 
 * 'QirPassRunner' class
 */
void QirPassRunner::append(std::string pass) {
    passes_.push_back(pass);
}

/* Applies all passes in the private vector 'passes_' to the 
 * QIR parsed into an LLVM module 'module'
 */
void /*PreservedAnalyses*/ QirPassRunner::run(Module &module, ModuleAnalysisManager &MAM) {
    // TODO HOW DO WE HANDLE 'PreservedAnalyses'?
    //PreservedAnalyses PA;

    while (!passes_.empty()) {
        // Get the name of the pass compiled as a shared library
        auto pass = passes_.back();

        // Load the library
		void *soHandle = dlopen(pass.c_str(), RTLD_LAZY);

        if(!soHandle) {
            std::cout << "[Pass Runner] Warning: Could not load shared library: " << pass << std::endl;
            passes_.pop_back();
            continue;
        }

        // Format the name of the pass and print it on screen
        size_t      lastSlash          = pass.find_last_of('/');
        std::string passName           = pass.substr(lastSlash + 4);
        size_t      lastDot            = passName.find_last_of('.');
        std::string passNameWithoutExt = passName.substr(0, lastDot);

        std::cout << "[Pass Runner] Applying pass: \033[1m" 
                  << passNameWithoutExt 
                  << "\033[0m" 
                  << std::endl;

        // Pointer to 'loadQirPass' function returning a pointer to the 'PassModule' object
        using passLoader = PassModule* (*)();

        // Dynamic loading and linking of the shared library
        passLoader loadQirPass = reinterpret_cast<passLoader>(dlsym(soHandle, "loadQirPass"));

        if(!loadQirPass) {
            std::cout << "[Pass Runner] Warning: Could not get factory function of pass: " 
                      << pass 
                      << std::endl;

            passes_.pop_back();
            dlclose(soHandle);
            continue;
        }

        PassModule *QirPass = loadQirPass();

        // Apply the pass to the LLVM module 'module'
        /*PA =*/ QirPass->run(module, MAM);

        // Free memory
        delete QirPass;
        dlclose(soHandle);

        passes_.pop_back();
    }

    //return PA;
}

