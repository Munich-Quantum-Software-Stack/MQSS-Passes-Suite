/**
 * @file QirRemoveNonEntrypointFunctions.cpp
 * @brief Implementation of the 'QirRemoveNonEntrypointFunctionsPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirRemoveNonEntrypointFunctions.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/RemoveNonEntrypointFunctions/RemoveNonEntrypointFunctionsPass.cpp
 */

#include <QirRemoveNonEntrypointFunctions.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirRemoveNonEntrypointFunctionsPass::run(
    Module &module, ModuleAnalysisManager &MAM, QDMI_Device dev)
{
    std::vector<Function *> functions_to_delete;

    for (auto &function : module)
        if (!function.isDeclaration() &&
            !function.hasFnAttribute("entry_point"))
            functions_to_delete.push_back(&function);

    for (auto &function : functions_to_delete)
        if (function->use_empty())
            function->eraseFromParent();

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the
 * 'QirRemoveNonEntrypointFunctionsPass' as a 'PassModule'.
 * @return QirRemoveNonEntrypointFunctionsPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirRemoveNonEntrypointFunctionsPass();
}
