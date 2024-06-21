/**
 * @file QirU3ToRzRyRzDecomposition.cpp
 * @brief Implementation of the 'QirU3ToRzRyRzDecompositionPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirU3Decomposition.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass decomposes a U3 gate into Rx and Ry gates.
 *
 * Adapted from: https://dl.acm.org/doi/10.5555/1972505
 *
 * @todo THERE SEEMS TO BE A POTENTIAL BUG WITHIN THIS PASS: ALL GATES ARE
 * REPLACED BY A U3 GATE
 */

#include <QirU3ToRzRyRzDecomposition.hpp>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses
QirU3ToRzRyRzDecompositionPass::run(Module &module, ModuleAnalysisManager &MAM,
                                    QDMI_Device dev)
{
    auto &Context = module.getContext();

    Function *functionKey = module.getFunction("__quantum__qis__U3__body");

    if (!functionKey)
        return PreservedAnalyses::all();

    Function *function =
        module.getFunction("__quantum__qis__U3_to_rzryrz__body");

    if (function)
        return PreservedAnalyses::all();

    Type *doubleType = Type::getDoubleTy(Context);
    StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    FunctionType *funcType = FunctionType::get(
        Type::getVoidTy(Context),
        {doubleType, doubleType, doubleType, qubitPtrType}, false);

    function = Function::Create(funcType, Function::ExternalLinkage,
                                "__quantum__qis__U3_to_rzryrz__body", module);

    BasicBlock *entryBlock = BasicBlock::Create(Context, "entry", function);
    IRBuilder<> builder(entryBlock);

    Function *qis_rz_body = module.getFunction("__quantum__qis__rz__body");
    Function *qis_ry_body = module.getFunction("__quantum__qis__ry__body");

    if (!qis_rz_body)
    {
        FunctionType *funcTypeRx = FunctionType::get(
            Type::getVoidTy(Context), {doubleType, qubitPtrType}, false);

        qis_rz_body = Function::Create(funcTypeRx, Function::ExternalLinkage,
                                       "__quantum__qis__rz__body", module);
    }

    if (!qis_ry_body)
    {
        FunctionType *funcTypeRy = FunctionType::get(
            Type::getVoidTy(Context), {doubleType, qubitPtrType}, false);

        qis_ry_body = Function::Create(funcTypeRy, Function::ExternalLinkage,
                                       "__quantum__qis__ry__body", module);
    }

    Value *a = function->getArg(0);
    Value *b = function->getArg(1);
    Value *c = function->getArg(2);

    Value *q = function->getArg(3);

    builder.CreateCall(qis_rz_body, {b, q});
    builder.CreateCall(qis_ry_body, {a, q});
    builder.CreateCall(qis_rz_body, {c, q});

    builder.CreateRetVoid();

    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();

    Function *functionValue =
        module.getFunction("__quantum__qis__U3_to_rzryrz__body");
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
 * @brief External function for loading the 'QirU3ToRzRyRzDecompositionPass' as
 * a 'PassModule'.
 * @return QirU3ToRzRyRzDecompositionPass
 */
extern "C" PassModule *loadQirPass()
{
    return new QirU3ToRzRyRzDecompositionPass();
}
