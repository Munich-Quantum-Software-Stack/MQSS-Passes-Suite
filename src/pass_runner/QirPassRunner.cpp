/**
 * @file QirPassRunner.cpp
 * @brief Implementation of the 'QirPassRunner' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/QirPassRunner.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 */

#include "QirPassRunner.hpp"
#include <algorithm>
#include <iostream>
#include <string>

using namespace llvm;

/**
 * @brief Default constructor of the 'QirPassRunner' class
 */
QirPassRunner::QirPassRunner() {}

/**
 * @brief Returns a reference to a 'QirPassRuner' object
 * @return QirPassRunner
 */
QirPassRunner &QirPassRunner::getInstance()
{
    static QirPassRunner instance;
    return instance;
}

/**
 * @brief Saves 'metadata' as the private metadata of the
 * 'QirPassRunner' class.
 * @param metadata The metadata attached to the module.
 */
void QirPassRunner::setMetadata(const QirMetadata &metadata)
{
    qirMetadata_ = metadata;
}

/**
 * @brief Empties all structures within the metadata.
 */
void QirPassRunner::clearMetadata()
{
    qirMetadata_.reversibleGates.clear();
    qirMetadata_.supportedGates.clear();
    qirMetadata_.availablePlatforms.clear();
    qirMetadata_.injectedAnnotations.clear();
}

/**
 * @brief Returns the private metadata of the 'QirPassRunner' class
 * @return QirMetadata
 */
QirMetadata &QirPassRunner::getMetadata() { return qirMetadata_; }

/**
 *  Inserts a pass in the private vector 'passes_' of the
 * 'QirPassRunner' class
 */
void QirPassRunner::append(std::string pass) { passes_.push_back(pass); }

/**
 * @brief  Applies all passes in the private vector 'passes_' to the
 * QIR parsed into an LLVM module 'module'
 * @param module The module of the submitted QIR.
 * @param MAM The module analysis manager.
 */
void /*PreservedAnalyses*/ QirPassRunner::run(Module &module,
                                              ModuleAnalysisManager &MAM)
{
    // TODO HOW DO WE HANDLE 'PreservedAnalyses'?
    // PreservedAnalyses PA;

    while (!passes_.empty())
    {
        // Get the name of the pass compiled as a shared library
        auto pass = passes_.back();

        // Load the library
        void *lib_handle = dlopen(pass.c_str(), RTLD_LAZY);

        if (!lib_handle)
        {
            std::cout << "   [Pass Runner].......Warning: Could not load "
                         "shared library: "
                      << pass << dlerror() << std::endl;

            passes_.pop_back();
            continue;
        }

        // Format the name of the pass and print it on screen
        size_t lastSlash = pass.find_last_of('/');
        std::string passName = pass.substr(lastSlash + 4);
        size_t lastDot = passName.find_last_of('.');
        std::string passNameWithoutExt = passName.substr(0, lastDot);

        std::cout << "   [Pass Runner].......Applying pass: "
                  << passNameWithoutExt << std::endl;

        // Pointer to 'loadQirPass' function returning a pointer to the
        // 'PassModule' object
        using passLoader = PassModule *(*)();

        // Dynamic loading and linking of the shared library
        passLoader loadQirPass =
            reinterpret_cast<passLoader>(dlsym(lib_handle, "loadQirPass"));

        if (!loadQirPass)
        {
            std::cout << "   [Pass Runner].......Warning: Could not get "
                         "factory function "
                         "of pass: "
                      << pass << std::endl;

            passes_.pop_back();
            dlclose(lib_handle);
            continue;
        }

        PassModule *QirPass = loadQirPass();

        // Apply the pass to the LLVM module 'module'
        /*PA =*/QirPass->run(module, MAM);

        // Free memory
        delete QirPass;
        dlclose(lib_handle);

        passes_.pop_back();
    }

    // return PA;
}
