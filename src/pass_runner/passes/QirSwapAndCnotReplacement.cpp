/**
 * @file QirSwapAndCnotReplacement.cpp
 * @brief Implementation of the 'QirSwapAndCnotReplacementPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirSwapAndCnotReplacement.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass removes two sequential Cnots acting on the same qubit.
 */

#include "../headers/QirSwapAndCnotReplacement.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirSwapAndCnotReplacementPass::run(Module &module,
                                   ModuleAnalysisManager & /*MAM*/)
{
    auto &Context = module.getContext();

    for (auto &function : module)
    {
        std::vector<CallInst *> gatesToRemove;
        std::vector<CallInst *> gatesToLeave;

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
                                            "__quantum__qis__swap__body")
                                        {
                                            // If to check if Swap and Cnot are
                                            // acting on the same qubits
                                            if ((prev_instruction
                                                         ->getArgOperand(0) ==
                                                     current_instruction
                                                         ->getArgOperand(0) &&
                                                 prev_instruction
                                                         ->getArgOperand(1) ==
                                                     current_instruction
                                                         ->getArgOperand(1)) ||
                                                (prev_instruction
                                                         ->getArgOperand(0) ==
                                                     current_instruction
                                                         ->getArgOperand(1) &&
                                                 prev_instruction
                                                         ->getArgOperand(1) ==
                                                     current_instruction
                                                         ->getArgOperand(0)))
                                            {
                                                gatesToRemove.push_back(
                                                    prev_instruction);
                                                gatesToLeave.push_back(
                                                    current_instruction);

                                                errs()
                                                    << "[Pass]..............."
                                                       "Replacing sequential"
                                                       "SWAP and CNOT\n";
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
        while (!gatesToRemove.empty())
        {
            auto *gateToRemove = gatesToRemove.back();
            gateToRemove->eraseFromParent();
            gatesToRemove.pop_back();
            auto *originalCnot = gatesToLeave.back();
            Function *newCnot =
                module.getFunction("__quantum__qis__cnot__body");
            if (!newCnot)
            {
                StructType *qubitType =
                    StructType::getTypeByName(Context, "Qubit");
                PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
                FunctionType *funcType =
                    FunctionType::get(Type::getVoidTy(Context),
                                      {qubitPtrType, qubitPtrType}, false);
                newCnot =
                    Function::Create(funcType, Function::ExternalLinkage,
                                     "__quantum__qis__cnot__body", module);
            }
            CallInst *newCnotInst =
                CallInst::Create(newCnot, {originalCnot->getOperand(1),
                                           originalCnot->getOperand(0)});
            newCnotInst->insertAfter(originalCnot);
            gatesToLeave.pop_back();
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirSwapAndCnotReplacementPass' as a
 * 'PassModule'.
 * @return QirSwapAndCnotReplacementPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirSwapAndCnotReplacementPass();
}
