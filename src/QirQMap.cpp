/**
 * @file QirQMap.cpp
 * @brief TODO
 */

#include <QirQMap.hpp>

#include <iostream>
#include <string>

using namespace llvm;

/**
 * @brief TODO
 *
 * @param TODO
 * @param TODO
 * @param TODO
 * @return PreservedAnalyses
 */
PreservedAnalyses QirQMapPass::run(
    Module &module, ModuleAnalysisManager & /*MAM*/, QDMI_Device dev)
{
    LLVMContext &Context = module.getContext();
    IRBuilder<> builder(Context);
    std::vector<Function *> functionsToDelete;
    std::string entryFunctionName;

    // Parse LLVM::Module to QC::QuantumComputation
    auto arch = mqt::createArchitecture(dev);
    auto qc   = qc::QuantumComputation(arch.getNqubits(), arch.getNqubits());

    for (auto &function : module)
    {
        for (auto &block : function)
        {
            for (auto &instruction : block)
            {
                if (auto *call_instr = dyn_cast<CallBase>(&instruction))
                {
                    if (auto *f = call_instr->getCalledFunction())
                    {
                        auto name = static_cast<std::string>(f->getName().str());
                        
                        if (name == "__quantum__qis__rx__body"
                         || name == "__quantum__qis__ry__body"
                         || name == "__quantum__qis__rz__body")
                        {
                            int qubit = -1;
                            double angle = -1.0;

                            Value *farg = call_instr->getArgOperand(0);

                            if (auto *const_fp = dyn_cast<ConstantFP>(farg))
                            {
                                APFloat val = const_fp->getValueAPF();
                                angle = val.convertToDouble();
                            }

                            if (auto *call_instru = dyn_cast<CallInst>(&instruction))
                            {
                                Value *qarg = call_instru->getArgOperand(1);

                                if (isa<ConstantPointerNull>(qarg)) 
                                    qubit = 0;
                                else if(ConstantExpr *constExpr = dyn_cast<ConstantExpr>(qarg))
                                {
                                    IntToPtrInst* castInstruction = dyn_cast<IntToPtrInst>(constExpr->getAsInstruction());
                                    ConstantInt *qubitInt = dyn_cast<ConstantInt>(
                                        castInstruction->getOperand(0)
                                    );
                                    qubit = qubitInt->getSExtValue();
                                }
                            }

                            if (angle == -1.0 || qubit == -1)
                            {
                                errs() << "   [Pass]................Warning: "
                                       << "ill-formed gate\n";
                                continue;
                            }

                            if (name == "__quantum__qis__rx__body")
                                qc.rx(angle, qubit);
                            else if (name == "__quantum__qis__ry__body")
                                qc.ry(angle, qubit);
                            else if (name == "__quantum__qis__rz__body")
                                qc.rz(angle, qubit);
                        }
                        else if (name == "__quantum__qis__cx__body" 
                              || name == "__quantum__qis__cnot__body"
                              || name == "__quantum__qis__cy__body"
                              || name == "__quantum__qis__cz__body")
                        {
                            int qubit_control = -1;
                            int qubit_target  = -1;

                            if (auto *call_instru = dyn_cast<CallInst>(&instruction))
                            {
                                Value *qarg_control = call_instru->getArgOperand(0);
                                Value *qarg_target  = call_instru->getArgOperand(1);

                                if (isa<ConstantPointerNull>(qarg_control))
                                    qubit_control = 0;
                                else if(ConstantExpr *constExpr = dyn_cast<ConstantExpr>(qarg_control))
                                {
                                    IntToPtrInst* castInstruction = dyn_cast<IntToPtrInst>(constExpr->getAsInstruction());
                                    ConstantInt *qubitInt = dyn_cast<ConstantInt>(
                                    castInstruction->getOperand(0));
                                    qubit_control = qubitInt->getSExtValue();
                                }

                                if (isa<ConstantPointerNull>(qarg_target))
                                    qubit_target = 0;
                                else if(ConstantExpr *constExpr = dyn_cast<ConstantExpr>(qarg_target))
                                {
                                    IntToPtrInst* castInstruction = dyn_cast<IntToPtrInst>(constExpr->getAsInstruction());
                                    ConstantInt *qubitInt = dyn_cast<ConstantInt>(
                                    castInstruction->getOperand(0));
                                    qubit_target = qubitInt->getSExtValue();
                                }
                            }

                            if (qubit_control == -1 || qubit_target == -1)
                            {
                                errs() << "   [Pass]................Warning: "
                                       << "ill-formed gate\n";
                                continue;
                            }

                            if (name == "__quantum__qis__cx__body" || name == "__quantum__qis__cnot__body")
                                qc.cx(qubit_control, qubit_target);
                            else if (name == "__quantum__qis__cy__body")
                                qc.cy(qubit_control, qubit_target);
                            else if (name == "__quantum__qis__cz__body")
                                qc.cz(qubit_control, qubit_target);
                        }
                        else if (name == "__quantum__qis__mz__body")
                            ; // QMap inserts a measure-all instruction
                        else if (name == "__quantum__qis__s__body"
                              || name == "__quantum__qis__t__body"
                              || name == "__quantum__qis__x__body"
                              || name == "__quantum__qis__y__body"
                              || name == "__quantum__qis__z__body"
                              || name == "__quantum__qis__h__body")
                        {
                            int qubit = -1;

                            if (auto *call_instru = dyn_cast<CallInst>(&instruction))
                            {
                                Value *qarg = call_instru->getArgOperand(0);

                                if (isa<ConstantPointerNull>(qarg))
                                    qubit = 0;
                                else if(ConstantExpr *constExpr = dyn_cast<ConstantExpr>(qarg))
                                {
                                    IntToPtrInst* castInstruction = dyn_cast<IntToPtrInst>(constExpr->getAsInstruction());
                                    ConstantInt *qubitInt = dyn_cast<ConstantInt>(
                                    castInstruction->getOperand(0));
                                    qubit = qubitInt->getSExtValue();
                                }
                            }

                            if (qubit == -1)
                            {
                                errs() << "   [Pass]................Warning: "
                                       << "ill-formed gate\n";
                                continue;
                            }

                            if (name == "__quantum__qis__s__body")
                                qc.s(qubit);
                            else if (name == "__quantum__qis__t__body")
                                qc.t(qubit);
                            else if (name == "__quantum__qis__x__body")
                                qc.x(qubit);
                            else if (name == "__quantum__qis__y__body")
                                qc.y(qubit);
                            else if (name == "__quantum__qis__z__body")
                                qc.z(qubit);
                            else if (name == "__quantum__qis__h__body")
                                qc.h(qubit);
                        }
                        else
                        {
                            errs() << "   [Pass]................Warning: "
                                   << "gate not supported: "
                                   << name << "\n";
                            continue;
                        }
                    }
                }
            }
        }

        functionsToDelete.push_back(&function);
    }

    // Map the circuit
    auto mapper = HeuristicMapper(qc, arch);
    mapper.map({});

    // Delete all the non-entry point functions
    // and delete the body of the entry point
    for (auto &function : functionsToDelete)
    {
        function->deleteBody();

        if (!function->hasFnAttribute("entry_point"))
            function->eraseFromParent();
        else
            entryFunctionName = function->getName().str();
    }

    // Fetch the entry point function
    Function *entryFunction = module.getFunction(entryFunctionName);
    assert(entryFunction);

    // Create again the entry block
    BasicBlock *entryBlock  = BasicBlock::Create(
        Context, 
        "entry", 
        entryFunction
    );

    builder.SetInsertPoint(entryBlock);

    // Parse from QC::QuantumComputation back to LLVM::Module

    // Insert initialize instruction
    FunctionType *initFuncType = FunctionType::get(
        Type::getVoidTy(Context), 
        {Type::getInt8PtrTy(Context)}, 
        false
    );

    Constant *NullPtr = ConstantPointerNull::get(Type::getInt8PtrTy(Context));
    Value *PtrArg = builder.CreateIntToPtr(NullPtr, Type::getInt8PtrTy(Context));
    
    Function *initFunction = module.getFunction("__quantum__rt__initialize");

    if (!initFunction)
    {
        initFunction = Function::Create(
            initFuncType,
            Function::ExternalLinkage,
            "__quantum__rt__initialize", 
            module
        );
    }

    builder.CreateCall(
        initFunction, 
        {PtrArg}
    );

    // Insert quantum gates
    StructType *qubitType  = StructType::getTypeByName(Context, "Qubit");
    StructType *resultType = StructType::getTypeByName(Context, "Result");

    PointerType *qubitPtrType  = PointerType::getUnqual(qubitType);
    PointerType *resultPtrType = PointerType::getUnqual(resultType);

    FunctionType *singleQubitFuncType = FunctionType::get(
        Type::getVoidTy(Context), 
        {qubitPtrType}, 
        false
    );

    FunctionType *twoQubitFuncType = FunctionType::get(
        Type::getVoidTy(Context), 
        {qubitPtrType, qubitPtrType}, 
        false
    );

    FunctionType *irreversibleFuncType = FunctionType::get(
        Type::getVoidTy(Context), 
        {qubitPtrType, resultPtrType}, 
        false
    );

    for (const auto& op : qc)
    {
        auto &targets  = op->getTargets();
        auto &controls = op->getControls();

        Function *newFunction;

        if (targets.size() == 1 && controls.size() == 1)
        {
            Value *target = ConstantInt::get(
                Type::getInt64Ty(Context), 
                targets[0]
            );

            Value *targetPtr = builder.CreateIntToPtr(
                target, 
                qubitPtrType
            );

            Value *control = ConstantInt::get(
                Type::getInt64Ty(Context),
                controls.begin()->qubit
            );

            Value *controlPtr = builder.CreateIntToPtr(
                control,
                qubitPtrType
            );

            switch (op->getType())
            {
                case qc::X:
                    newFunction = module.getFunction("__quantum__qis__cx__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            twoQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__cx__body", 
                            module
                        );
                    }
                    break;
                case qc::Y:
                    newFunction = module.getFunction("__quantum__qis__cy__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            twoQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__cy__body", 
                            module
                        );
                    }
                    break;
                case qc::Z:
                    newFunction = module.getFunction("__quantum__qis__cz__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            twoQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__cz__body",
                            module
                        );
                    }
                    break;
            } // switch (op->getType())

            if (newFunction)
                builder.CreateCall(
                    newFunction, 
                    {controlPtr, targetPtr}
                );

        } // targets.size() == 1 && controls.size() == 1
        else if (targets.size() == 1 && controls.size() == 0)
        {
            Value *target = ConstantInt::get(
                Type::getInt64Ty(Context), 
                targets[0]
            );

            Value *targetPtr = builder.CreateIntToPtr(
                target, 
                qubitPtrType
            );

            switch (op->getType())
            {
                case qc::X:
                    newFunction = module.getFunction("__quantum__qis__x__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__x__body", 
                            module
                        );
                    }
                    break;
                case qc::Y:
                    newFunction = module.getFunction("__quantum__qis__y__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__y__body", 
                            module
                        );
                    }
                    break;
                case qc::Z:
                    newFunction = module.getFunction("__quantum__qis__z__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__z__body", 
                            module
                        );
                    }
                    break;
                case qc::H:
                    newFunction = module.getFunction("__quantum__qis__h__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__h__body", 
                            module
                        );
                    }
                    break;
                case qc::S:
                    newFunction = module.getFunction("__quantum__qis__s__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__s__body", 
                            module
                        );
                    }
                    break;
                case qc::T:
                    newFunction = module.getFunction("__quantum__qis__t__body");

                    if (!newFunction)
                    {
                        newFunction = Function::Create(
                            singleQubitFuncType,
                            Function::ExternalLinkage,
                            "__quantum__qis__t__body", 
                            module
                        );
                    }
                    break;
            } // switch (op->getType())

            if (newFunction)
                builder.CreateCall(
                    newFunction, 
                    {targetPtr}
                );

        } // targets.size() == 1 && controls.size() == 0
    } // for (const auto& op : qc)

    // Insert measurements
    int i;
    for (i = 0; i < arch.getNqubits(); i++)
    {
        Value *qubitTarget = ConstantInt::get(
            Type::getInt64Ty(Context), 
            i
        );

        Value *qubitTargetPtr = builder.CreateIntToPtr(
            qubitTarget,
            qubitPtrType
        );

        Value *resultTarget = ConstantInt::get(
            Type::getInt64Ty(Context),
            i
        );

        Value *resultTargetPtr = builder.CreateIntToPtr(
            resultTarget,
            resultPtrType
        );

        Function *mzFunction = module.getFunction("__quantum__qis__mz__body");

        if (!mzFunction)
        {
            mzFunction = Function::Create(
                irreversibleFuncType,
                Function::ExternalLinkage,
                "__quantum__qis__mz__body",
                module
            );
        }

        builder.CreateCall(
            mzFunction, 
            {qubitTargetPtr, resultTargetPtr}
        );
    }

    // Insert return
    builder.CreateRet(ConstantInt::get(Type::getInt64Ty(Context), 0));

    return PreservedAnalyses::none();
}

/**
 * @brief TODO
 * @return QirQMapPass
 */
#ifdef __cplusplus
extern "C" 
{
#endif
PassModule *loadQirPass()
{
    return new QirQMapPass();
}
#ifdef __cplusplus
} // extern "C"
#endif
