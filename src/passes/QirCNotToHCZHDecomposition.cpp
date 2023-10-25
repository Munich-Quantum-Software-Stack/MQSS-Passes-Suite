/**
 * @file QirCNotToHCZHDecomposition.cpp
 * @brief Implementation of the 'QirCNotToHCZHDecompositionPass' class. <a href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirCNotToHCZHDecomposition.cpp?ref_type=heads">Go to the source code of this file.</a>
 *
 * Adapted from: https://quantumcomputing.stackexchange.com/a/13784
 */

#include "../headers/QirCNotToHCZHDecomposition.hpp"

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirCNotToHCZHDecompositionPass::run(Module &module, ModuleAnalysisManager &MAM) {
    auto& Context = module.getContext();

    Function *functionKey = module.getFunction("__quantum__qis__cnot__body");
    
    if (!functionKey)
        return PreservedAnalyses::all();

    Function *function = module.getFunction("__quantum__qis__cnot_to_hczh__body");

    if (function)
        return PreservedAnalyses::all();

    StructType  *qubitType    = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    FunctionType *funcType = FunctionType::get(
        Type::getVoidTy(Context),
        {
            qubitPtrType,
            qubitPtrType
        },
        false
    );

    function = Function::Create(
        funcType,
        Function::ExternalLinkage,
        "__quantum__qis__cnot_to_hczh__body",
        module
    );

    BasicBlock *entryBlock = BasicBlock::Create(Context, "entry", function);
    IRBuilder<> builder(entryBlock);

    Function *qis_h_body = module.getFunction("__quantum__qis__h__body");
    Function *qis_cz_body = module.getFunction("__quantum__qis__cz__body");

    if (!qis_h_body) {
        FunctionType *funcTypeH = FunctionType::get(
            Type::getVoidTy(Context),
            {
                qubitPtrType
            },
            false
        );

        qis_h_body = Function::Create(
            funcTypeH,
            Function::ExternalLinkage,
            "__quantum__qis__h__body",
            module
        );
    }

    if (!qis_cz_body) {
        FunctionType *funcTypeCz = FunctionType::get(
            Type::getVoidTy(Context),
            {
                qubitPtrType,
                qubitPtrType
            },
            false
        );

        qis_cz_body = Function::Create(
            funcTypeCz,
            Function::ExternalLinkage,
            "__quantum__qis__cz__body",
            module
        );
    }

    Value *p = function->getArg(0);
    Value *q = function->getArg(1);

    builder.CreateCall(
        qis_h_body,
        {q}
    );
    builder.CreateCall(
        qis_cz_body,    
        {p, q}
    );
    builder.CreateCall(
        qis_h_body,
        {q}
    );

    builder.CreateRetVoid();
    
    QirPassRunner &QPR = QirPassRunner::getInstance();
    QirMetadata &qirMetadata = QPR.getMetadata();

    Function *functionValue = module.getFunction("__quantum__qis__cnot_to_hczh__body");
    if (functionValue) {
        auto key   = static_cast<std::string>(functionKey->getName());
        auto value = static_cast<std::string>(functionValue->getName());
        qirMetadata.injectAnnotation(key, value);
        qirMetadata.setRemoveCallAttributes(false);
    }

    QPR.setMetadata(qirMetadata);    

    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirCNotToHCZHDecompositionPass' as a 'PassModule'.
 * @return QirCNotToHCZHDecompositionPass 
 */
extern "C" PassModule* loadQirPass() {
    return new QirCNotToHCZHDecompositionPass();
}
