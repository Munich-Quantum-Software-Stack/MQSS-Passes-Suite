/**
 * @file QirRedundantGatesCancellation.cpp
 * @brief Implementation of the 'QirRedundantGatesCancellationPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirRedundantGatesCancellation.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes redundant one-qubit gates, that is, equivalent gates
 * acting back to back on the same qubit.
 */

#include "../headers/QirRedundantGatesCancellation.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirRedundantGatesCancellationPass::run(Module &module,
                                       ModuleAnalysisManager & /*MAM*/)
{
    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();

    for (auto reversibleGate : qirMetadata.reversibleGates)
    {
        for (auto &function : module)
        {
            std::vector<CallInst *> gatesToRemove;
            std::vector<CallInst *> singletonContainer;
            for (auto &block : function)
            {
                for (auto &instruction : block)
                {
                    auto *current_instruction =
                        dyn_cast<CallInst>(&instruction);

                    if (current_instruction)
                    {
                        auto *current_function =
                            current_instruction->getCalledFunction();

                        if (current_function == nullptr)
                            continue;

                        std::string current_name = static_cast<std::string>(
                            current_function->getName());

                        if (current_name == reversibleGate)
                        {
                            if (singletonContainer.size() == 0)
                            {
                                singletonContainer.push_back(
                                    current_instruction);
                                continue;
                            }

                            CallInst *last_instruction =
                                singletonContainer.back();

                            gatesToRemove.push_back(last_instruction);
                            gatesToRemove.push_back(current_instruction);

                            errs() << "   [Pass]................Redundant gate "
                                      "pair found: "
                                   << reversibleGate << '\n';
                        }
                        singletonContainer.clear();
                    }
                }
            }

            assert(((void)"Number of gates to be removed is not even",
                    gatesToRemove.size() % 2 == 0));

            while (!gatesToRemove.empty())
            {
                auto *gateToRemove = gatesToRemove.back();
                gateToRemove->eraseFromParent();
                gatesToRemove.pop_back();
            }
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirRedundantGatesCancellationPass'
 * as a 'PassModule'.
 * @return QirRedundantGatesCancellationPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirRedundantGatesCancellationPass();
}
