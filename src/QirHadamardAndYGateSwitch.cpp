/**
 * @file QirHadamardAndYGateSwitch.cpp
 * @brief Implementation of the 'QirHadamardAndYGateSwitchPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirHadamardAndYGateSwitch.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from: https://dl-acm-org.translate.goog/doi/10.5555/1972505
 */

#include <QirHadamardAndYGateSwitch.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirHadamardAndYGateSwitchPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/, QDMI_Device dev)
{
    auto &Context = module.getContext();

    for (auto &function : module)
    {
        std::vector<CallInst *> currentGates;
        std::vector<CallInst *> previousGates;

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

                    if (current_name == "__quantum__qis__y__body")
                    {
                        if (prev_instruction)
                        {
                            auto *prev_function =
                                dyn_cast<CallInst>(prev_instruction)
                                    ->getCalledFunction();

                            if (prev_function)
                            {
                                std::string previous_name =
                                    prev_function->getName().str();

                                if (previous_name == "__quantum__qis__h__body")
                                {
                                    previousGates.push_back(prev_instruction);
                                    currentGates.push_back(current_instruction);
                                    errs() << "              Switching: "
                                           << previous_name << " and "
                                           << current_name << '\n';
                                }
                            }
                        }
                    }
                }
                prev_instruction = current_instruction;
            }
        }
        while (!currentGates.empty())
        {
            auto *currentGate = currentGates.back();
            auto *prevGate = previousGates.back();
            currentGate->moveBefore(prevGate);
            currentGates.pop_back();
            previousGates.pop_back();
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirHadamardAndYGateSwitchPass' as a
 * 'PassModule'.
 * @return QirHadamardAndYGateSwitchPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirHadamardAndYGateSwitchPass();
}
