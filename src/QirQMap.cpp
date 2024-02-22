/**
 * @file QirQMap.cpp
 * @brief TODO
 */

#include <QirQMap.hpp>
#include <iostream>

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
                                    castInstruction->getOperand(0));
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
                            ; // QMap inserts a measure all instruction
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
    }

    auto mapper = HeuristicMapper(qc, arch);
    mapper.map({});

    // TODO PARSE FROM QuantumComputation TO LLVM::Module

    mapper.dumpResult(std::cout, qc::Format::OpenQASM3);

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
