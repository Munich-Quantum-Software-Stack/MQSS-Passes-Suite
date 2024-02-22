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

    //for (auto& it : qc.ops)
    //{
    //    if (it->getType() != qc::X)
    //        errs() << "\n\tFound an X\n";
    //}

    // Empty LLVM::Module
    std::string const QIS_START = "__quantum__qis_";
    std::vector<Instruction *> instructionsToDelete;

    for (auto &function : module)
        for (auto &block : function)
            for (auto &instruction : block)
                if (auto *call_instr = dyn_cast<CallBase>(&instruction))
                    if (auto *f = call_instr->getCalledFunction())
                    {
                        auto name = static_cast<std::string>(f->getName().str());
                        bool is_quantum = (name.size() >= QIS_START.size() 
                                        && name.substr(0, QIS_START.size()) 
                                        == QIS_START);

                        if (is_quantum)
                            instructionsToDelete.push_back(&instruction);
                    }
    
    for (Instruction *instr : instructionsToDelete)
        instr->eraseFromParent();

    std::stringstream buffer;
    mapper.dumpResult(buffer, qc::Format::OpenQASM3);
    std::istringstream iss(buffer.str());
    std::string line;
    errs() << "\nCircuit:\n" << buffer.str() << "\n";
    while (std::getline(iss, line))
    {
        if (line.find("//")       == std::string::npos
         && line.find("OPENQASM") == std::string::npos
         && line.find("include")  == std::string::npos
         && line.find("bit")      == std::string::npos)
        {
            std::istringstream lineStream(line);
            std::string item;
            std::vector<std::string> items;

            while (std::getline(lineStream, item, ' '))
                if (!item.empty())
                    items.push_back(item);
            
            //if (items[0] == "cx")
            //    errs() << "\n\tFound cx\n";
        }
    }

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
