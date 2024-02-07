/**
 * @file QirSwapToCnotsDecomposition.cpp
 * @brief Implementation of the 'QirSwapToCnotsDecompositionPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirSwapToCnotsDecomposition.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from: https://threeplusone.com/pubs/on_gates.pdf -- 6.4 Swap gate
 */

#include <QirSwapToCnotsDecomposition.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirSwapToCnotsDecompositionPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/, QDMI_Device dev)
{
    auto &Context = module.getContext();

    Function *functionKey = module.getFunction("__quantum__qis__swap__body");

    if (!functionKey)
        return PreservedAnalyses::all();

    Function *function =
        module.getFunction("__quantum__qis__swap_to_cnots__body");

    if (function)
        return PreservedAnalyses::all();

    StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    FunctionType *funcType = FunctionType::get(
        Type::getVoidTy(Context), {qubitPtrType, qubitPtrType}, false);

    function = Function::Create(funcType, Function::ExternalLinkage,
                                "__quantum__qis__swap_to_cnots__body", module);

    BasicBlock *entryBlock = BasicBlock::Create(Context, "entry", function);
    IRBuilder<> builder(entryBlock);

    Function *qis_cnot_body = module.getFunction("__quantum__qis__cnot__body");

    if (!qis_cnot_body)
    {
        FunctionType *funcTypeCnot = FunctionType::get(
            Type::getVoidTy(Context), {qubitPtrType, qubitPtrType}, false);

        qis_cnot_body =
            Function::Create(funcTypeCnot, Function::ExternalLinkage,
                             "__quantum__qis__cnot__body", module);
    }

    Value *p = function->getArg(0);
    Value *q = function->getArg(1);

    builder.CreateCall(qis_cnot_body, {p, q});
    builder.CreateCall(qis_cnot_body, {q, p});
    builder.CreateCall(qis_cnot_body, {p, q});

    builder.CreateRetVoid();

    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();

    Function *functionValue =
        module.getFunction("__quantum__qis__swap_to_cnots__body");
    if (functionValue)
    {
        auto key = static_cast<std::string>(functionKey->getName());
        auto value = static_cast<std::string>(functionValue->getName());
        qirMetadata.injectAnnotation(key, value);
        qirMetadata.setRemoveCallAttributes(false);
    }

    QPR.setMetadata(qirMetadata);

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirSwapToCnotsDecompositionPass' as
 * a 'PassModule'.
 * @return QirSwapToCnotsDecompositionPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirSwapToCnotsDecompositionPass();
}
