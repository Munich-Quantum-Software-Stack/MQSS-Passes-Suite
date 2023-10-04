#include "../headers/QirU3Decomposition.hpp"

using namespace llvm;

PreservedAnalyses QirU3DecompositionPass::run(Module &module, ModuleAnalysisManager &MAM) {
    auto& Context = module.getContext();

    Function *functionKey = module.getFunction("__quantum__qis__U3__body");
    
    if (!functionKey)
        return PreservedAnalyses::all();

    Function *function = module.getFunction("__quantum__qis__rxryrx__body");

    if (function)
        return PreservedAnalyses::all();

    Type        *doubleType   = Type::getDoubleTy(Context);
    StructType  *qubitType    = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    FunctionType *funcType = FunctionType::get(
        Type::getVoidTy(Context),
        {
            doubleType,
            doubleType,
            doubleType,
            qubitPtrType
        },
        false
    );

    function = Function::Create(
        funcType,
        Function::ExternalLinkage,
        "__quantum__qis__rxryrx__body",
        module/*.get()*/
    );

    BasicBlock *entryBlock = BasicBlock::Create(Context, "entry", function);
    IRBuilder<> builder(entryBlock);

    Function *qis_rx_body = module.getFunction("__quantum__qis__rx__body");
    Function *qis_ry_body = module.getFunction("__quantum__qis__ry__body");

    if (!qis_rx_body) {
        FunctionType *funcTypeRx = FunctionType::get(
            Type::getVoidTy(Context),
            {
                doubleType,
                qubitPtrType
            },
            false
        );

        qis_rx_body = Function::Create(
            funcTypeRx,
            Function::ExternalLinkage,
            "__quantum__qis__rx__body",
            module/*.get()*/
        );
    }

    if (!qis_ry_body) {
        FunctionType *funcTypeRy = FunctionType::get(
            Type::getVoidTy(Context),
            {
                doubleType,
                qubitPtrType
            },
            false
        );

        qis_ry_body = Function::Create(
            funcTypeRy,
            Function::ExternalLinkage,
            "__quantum__qis__ry__body",
            module/*.get()*/
        );
    }

    Value *a = function->getArg(0);
    Value *b = function->getArg(1);
    Value *c = function->getArg(2);

    Value *q = function->getArg(3);

    Value *sum_ab         = builder.CreateFAdd(a, b);
    Value *sum_bc         = builder.CreateFAdd(b, c);
    Value *prod_bc        = builder.CreateFMul(b, c);
    Value *sum_ab_prod_bc = builder.CreateFAdd(sum_ab, prod_bc);

    builder.CreateCall(
        qis_rx_body,
        {sum_ab, q}
    );
    builder.CreateCall(
        qis_ry_body,    
        {sum_bc, q}
    );
    builder.CreateCall(
        qis_rx_body,
        {sum_ab_prod_bc, q}
    );

    builder.CreateRetVoid();

    // XXX THIS IS HOW YOU APPEND METADATA TO THE MODULE'S CONTEXT
    // (These metadata will NOT be attached to the module's IR)
    QirMetadata &qirMetadata = QirPassRunner::getInstance().getMetadata();

    Function *functionValue = module.getFunction("__quantum__qis__rxryrx__body");
    if (functionValue) {
        auto key   = static_cast<std::string>(functionKey->getName());
        auto value = static_cast<std::string>(functionValue->getName());
        qirMetadata.injectAnnotation(key, value);
        qirMetadata.setRemoveCallAttributes(false);
    }

    QirPassRunner::getInstance().setMetadata(qirMetadata);    

    return PreservedAnalyses::none();
}

extern "C" PassModule* createQirPass() {
    return new QirU3DecompositionPass();
}
