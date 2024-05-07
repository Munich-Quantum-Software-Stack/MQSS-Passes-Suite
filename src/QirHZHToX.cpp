/**
 * @file QirHZHToX.cpp
 * @brief Implementation of the 'QirHZHToXPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirHZHToX.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from: https://threeplusone.com/pubs/on_gates.pdf -- 3.5 Hadamard
 * Gates
 *
 */

#include <QirHZHToX.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 *
 */
PreservedAnalyses QirHZHToXPass::run(Module &module,
    ModuleAnalysisManager & /*MAM*/)
{
    auto &Context = module.getContext();

    std::unordered_set<Value *> qubits;
    StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    for (auto &function : module)
    {
        for (auto &block : function)
        {
            for (auto &instruction : block)
            {
                auto *current_instruction = dyn_cast<CallInst>(&instruction);

                if (current_instruction)
                {
                    for (auto &use : current_instruction->args())
                    {
                        Value *operand = use;
                        if (operand->getType() == qubitPtrType)
                        {
                            qubits.insert(operand);
                        }
                    }
                }
            }
        }
    }

    std::vector<CallInst *> instructions;
    std::vector<CallInst *> instructionsToRemove;
    std::vector<CallInst *> instructionsToReplaceWithX;

    for (auto &qubit : qubits)
    {
        for (auto user : qubit->users())
        {
            CallInst *instruction = dyn_cast_or_null<CallInst>(user);

            if (!instruction)
            {
                continue;
            }

            auto *function = instruction->getCalledFunction();

            if (function == nullptr)
            {
                continue;
            }

            std::string name = function->getName().str();

            if (instructions.size() == 0 && name == "__quantum__qis__h__body")
            {
                instructions.push_back(instruction);
                continue;
            }

            if (instructions.size() == 1)
            {
                if (name == "__quantum__qis__z__body")
                {
                    instructions.push_back(instruction);
                    continue;
                }
                else if (name == "__quantum__qis__h__body")
                {
                    instructions.pop_back();
                    instructions.push_back(instruction);
                    continue;
                }
            }

            if (instructions.size() == 2 && name == "__quantum__qis__h__body")
            {
                instructionsToReplaceWithX.push_back(instruction);
                instructionsToRemove.push_back(instructions[0]);
                instructionsToRemove.push_back(instructions[1]);
            }

            instructions.clear();
        }
    }

    Function *newX = module.getFunction("__quantum__qis__z__body");
    if (!newX)
    {
        FunctionType *funcType =
            FunctionType::get(Type::getVoidTy(Context), {qubitPtrType}, false);
        newX = Function::Create(funcType, Function::ExternalLinkage,
                                "__quantum__qis__z__body", module);
    }

    while (!instructionsToReplaceWithX.empty())
    {
        auto *gateToReplace = instructionsToReplaceWithX.back();
        CallInst *newInst =
            CallInst::Create(newX, {gateToReplace->getOperand(0)});
        ReplaceInstWithInst(gateToReplace, newInst);
        instructionsToReplaceWithX.pop_back();
    }

    while (!instructionsToRemove.empty())
    {
        auto *gateToRemove = instructionsToRemove.back();
        gateToRemove->eraseFromParent();
        instructionsToRemove.pop_back();
    }

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirHZHToXPass' as a
 * 'PassModule'.
 * @return QirHZHToXPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirHZHToXPass();
}
