/**
 * @file QirReverseCnot.cpp
 * @brief Implementation of the 'QirReverseCnotPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirReverseCnot.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://agra.informatik.uni-bremen.de/doc/konf/2021_DSD_CNOTs_remote_gates.pdf
 * Fig.1
 */

#include <QirReverseCnot.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirReverseCnotPass::run(Module &module,
                                          ModuleAnalysisManager & /*MAM*/)
{
    auto &Context = module.getContext();

    for (auto &function : module)
    {
        std::vector<CallInst *> cnotsToReverse;

        for (auto &block : function)
        {
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
                        cnotsToReverse.push_back(current_instruction);
                        errs() << "   [Pass]................Reversing Cnot\n";
                    }
                }
            }
        }

        Function *newCnot = module.getFunction("__quantum__qis__cnot__body");
        Function *newH = module.getFunction("__quantum__qis__h__body");
        if (!newH)
        {
            StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
            PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
            FunctionType *funcType = FunctionType::get(Type::getVoidTy(Context),
                                                       {qubitPtrType}, false);
            newH = Function::Create(funcType, Function::ExternalLinkage,
                                    "__quantum__qis__h__body", module);
        }

        while (!cnotsToReverse.empty())
        {
            auto *cnotToReverse = cnotsToReverse.back();
            CallInst *newCnotInst =
                CallInst::Create(newCnot, {cnotToReverse->getOperand(1),
                                           cnotToReverse->getOperand(0)});
            CallInst *newBeforeControlHInst =
                CallInst::Create(newH, {cnotToReverse->getOperand(1)});
            CallInst *newBeforeTargetHInst =
                CallInst::Create(newH, {cnotToReverse->getOperand(0)});
            CallInst *newAfterControlHInst =
                CallInst::Create(newH, {cnotToReverse->getOperand(1)});
            CallInst *newAfterTargetHInst =
                CallInst::Create(newH, {cnotToReverse->getOperand(0)});
            newBeforeControlHInst->insertBefore(cnotToReverse);
            newBeforeTargetHInst->insertBefore(cnotToReverse);
            newAfterControlHInst->insertAfter(cnotToReverse);
            newAfterTargetHInst->insertAfter(cnotToReverse);
            ReplaceInstWithInst(cnotToReverse, newCnotInst);
            cnotsToReverse.pop_back();
        }
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirReverseCnotPass' as a
 * 'PassModule'.
 * @return QirReverseCnotPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirReverseCnotPass();
}
