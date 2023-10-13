#include "../headers/QirRzDecomposition.hpp"

using namespace llvm;

PreservedAnalyses QirRzDecompositionPass::run(Module &module, ModuleAnalysisManager &MAM) {
    auto& Context = module.getContext();

    Function *functionKey = module.getFunction("__quantum__qis__rz__body");
    
    if (!functionKey)
        return PreservedAnalyses::all();

    Function *function = module.getFunction("__quantum__qis__rz_to_rxryrx__body");

    if (function)
        return PreservedAnalyses::all();

    Type        *doubleType   = Type::getDoubleTy(Context);
    StructType  *qubitType    = StructType::getTypeByName(Context, "Qubit");
    PointerType *qubitPtrType = PointerType::getUnqual(qubitType);

    FunctionType *funcType = FunctionType::get(
        Type::getVoidTy(Context),
        {
            doubleType,
            qubitPtrType
        },
        false
    );

    function = Function::Create(
        funcType,
        Function::ExternalLinkage,
        "__quantum__qis__rz_to_rxryrx__body",
        module
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
            module
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
            module
        );
    }

    Value *a = function->getArg(0);
    Value *q = function->getArg(1);

    const double pi_div_2       =  3.14159265358979323846 / 2.0;
    const double minus_pi_div_2 = -3.14159265358979323846 / 2.0;

    Value *pi_over_2       = ConstantFP::get(Context, APFloat(pi_div_2));
    Value *minus_pi_over_2 = ConstantFP::get(Context, APFloat(minus_pi_div_2));

    builder.CreateCall(
        qis_rx_body,
        {pi_over_2, q}
    );
    builder.CreateCall(
        qis_ry_body,    
        {a, q}
    );
    builder.CreateCall(
        qis_rx_body,
        {minus_pi_over_2, q}
    );

    builder.CreateRetVoid();

    QirMetadata &qirMetadata = QirPassRunner::getInstance().getMetadata();

    Function *functionValue = module.getFunction("__quantum__qis__rz_to_rxryrx__body");
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
    return new QirRzDecompositionPass();
}
