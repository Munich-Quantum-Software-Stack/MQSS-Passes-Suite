/**
 * @file QirYGateAndHadamardSwitch.cpp
 * @brief Implementation of the 'QirYGateAndHadamardSwitchPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirYGateAndHadamardSwitch.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from: https://dl.acm.org/doi/10.5555/1972505
 */

#include <QirYGateAndHadamardSwitch.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirYGateAndHadamardSwitchPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/, QDMI_Device dev)
{
    auto &Context = module.getContext();

    for (auto &function : module)
    {
        std::vector<CallInst *> previousGates;
        std::vector<CallInst *> currentGates;

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

                    if (current_name == "__quantum__qis__h__body")
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

                                if (previous_name == "__quantum__qis__y__body")
                                {
                                    currentGates.push_back(current_instruction);
                                    previousGates.push_back(prev_instruction);
                                    errs() << "                Switching: "
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
        while (!previousGates.empty())
        {
            auto *prevGate = previousGates.back();
            auto *currentGate = currentGates.back();
            currentGate->moveBefore(prevGate);
            previousGates.pop_back();
            currentGates.pop_back();
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirYGateAndHadamardSwitchPass' as a
 * 'PassModule'.
 * @return QirYGateAndHadamardSwitchPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirYGateAndHadamardSwitchPass();
}
