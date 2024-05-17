/**
 * @file QirDoubleCnotCancellation.cpp
 * @brief Implementation of the 'QirDoubleCnotCancellationPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirDoubleCnotCancellation.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes two sequential Cnots acting on the same qubit.
 *
 */

#include <QirDoubleCnotCancellation.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirDoubleCnotCancellationPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/)
{
    for (auto &function : module)
    {
        std::vector<CallInst *> gatesToRemove;
        for (auto &block : function)
        {
            CallInst *prev_instruction = nullptr;

            for (auto &instruction : block)
            {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if (current_instruction)
                {
                    auto *current_function =
                        current_instruction->getCalledFunction();

                    if (current_function == nullptr)
                        continue;

                    std::string current_name =
                        current_function->getName().str();

                    if (current_name == "__quantum__qis__cnot__body")
                    {
                        if (prev_instruction)
                        {
                            if (auto *callInst =
                                    dyn_cast<CallInst>(prev_instruction))
                            {
                                if (auto *prev_function =
                                        callInst->getCalledFunction())
                                {
                                    if (prev_function)
                                    {
                                        std::string previous_name =
                                            prev_function->getName().str();

                                        if (previous_name ==
                                            "__quantum__qis__cnot__body")
                                        {
                                            if (prev_instruction->getArgOperand(
                                                    0) ==
                                                    current_instruction
                                                        ->getArgOperand(0) &&
                                                prev_instruction->getArgOperand(
                                                    1) ==
                                                    current_instruction
                                                        ->getArgOperand(1))
                                            {
                                                gatesToRemove.push_back(
                                                    prev_instruction);
                                                gatesToRemove.push_back(
                                                    current_instruction);

                                                prev_instruction = nullptr;
                                                errs()
                                                    << "   "
                                                       "[Pass]..............A "
                                                       "pair of Cnot gates "
                                                       "found\n";
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                prev_instruction = current_instruction;
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
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirDoubleCnotCancellationPass' as a
 * 'PassModule'.
 * @return QirDoubleCnotCancellationPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirDoubleCnotCancellationPass();
}
