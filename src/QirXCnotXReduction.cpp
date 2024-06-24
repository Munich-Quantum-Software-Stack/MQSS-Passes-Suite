/**
 * @file QirXCnotXReduction.cpp
 * @brief Implementation of the 'QirXCnotXReductionPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirXCnotXReduction.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes X gates surrounding a CNOT gate.
 *
 * Adapted from: https://link.springer.com/chapter/10.1007/978-981-287-996-7_2
 */

#include <iostream>
#include <QirXCnotXReduction.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirXCnotXReductionPass::run(Module &module,
                                              ModuleAnalysisManager & /*MAM*/)
{
    for (auto &function : module)
    {
        for (auto &block : function)
        {
            std::vector<CallInst *> sought_sequence;
            std::vector<CallInst *> instructions_to_remove;

            for (auto &instruction : block)
            {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if (!current_instruction)
                {
                    sought_sequence.clear();
                    continue;
                }

                auto *current_function =
                    current_instruction->getCalledFunction();

                if (current_function == nullptr)
                {
                    sought_sequence.clear();
                    continue;
                }

                std::string current_name = current_function->getName().str();

                if (current_name == "__quantum__qis__cnot__body" &&
                    sought_sequence.size() == 1)
                {
                    auto *prev_instruction = sought_sequence.back();

                    assert(
                        ((void)"An error was encountered during gate removal",
                         prev_instruction));

                    auto *prev_function = prev_instruction->getCalledFunction();

                    assert(
                        ((void)"An error was encountered during gate removal",
                         prev_function != nullptr));

                    std::string prev_name = prev_function->getName().str();

                    if (prev_name == "__quantum__qis__x__body")
                        sought_sequence.push_back(current_instruction);
                    else
                        sought_sequence.clear();
                }
                else if (current_name == "__quantum__qis__x__body")
                {
                    if (sought_sequence.size() == 1) {
                        sought_sequence.clear();
                    }
                    if (sought_sequence.empty())
                    {
                        sought_sequence.push_back(current_instruction);
                        continue;
                    }

                    
                    auto *prev_instruction = sought_sequence.back();
                    sought_sequence.pop_back();


                    auto *prev_prev_instruction = sought_sequence.back();
                    sought_sequence.pop_back();

                    Value *x_1_arg = prev_prev_instruction->getArgOperand(0);
                    Value *cnot_arg = prev_instruction->getArgOperand(1);
                    Value *x_2_arg = current_instruction->getArgOperand(0);

                    if (x_1_arg == cnot_arg && cnot_arg == x_2_arg)
                    {
                        instructions_to_remove.push_back(prev_prev_instruction);
                        instructions_to_remove.push_back(current_instruction);
                    }
                    else
                        sought_sequence.clear();
                }
                else
                {
                    sought_sequence.clear();
                }
            }

            while (!instructions_to_remove.empty())
            {
                auto *instruction_to_remove = instructions_to_remove.back();
                instruction_to_remove->eraseFromParent();
                instructions_to_remove.pop_back();
            }
        }
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirXCnotXReductionPass' as a
 * 'PassModule'.
 * @return QirXCnotXReductionPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirXCnotXReductionPass();
}
