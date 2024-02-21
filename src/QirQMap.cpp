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
    //auto &Context = module.getContext();
    //StructType *qubitType = StructType::getTypeByName(Context, "Qubit");
    //PointerType *qubitPtrType = PointerType::getUnqual(qubitType);
    //ConstantPointerNull *nullQubitPointer =
    //    ConstantPointerNull::get(qubitPtrType);

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
                        
                        if (name == "__quantum__qis__rx__body")
                        {
                            Value *farg = call_instr->getArgOperand(0);
                            Value *qarg = call_instr->getArgOperand(1);
                            auto *angle = dyn_cast_or_null<ConstantFP>(farg);

                            //if (auto *inttoptr_inst = dyn_cast<IntToPtrInst>(qarg))
                            //    if (auto *constant_int = dyn_cast<ConstantInt>(inttoptr_inst->getOperand(0)))
                            //        int arg_value = constant_int->getSExtValue();

                            errs() << "\n\tangle: " << angle << "\n";

                            qc.rx(0.0, 0);
                        }
                        else if (name == "__quantum__qis__ry__body")
                            qc.ry(0.0, 0);
                        else if (name == "__quantum__qis__rz__body")
                            qc.rz(0.0, 0);
                        else if (name == "__quantum__qis__cx__body" || name == "__quantum__qis__cnot__body")
                            qc.cx(0, 1);
                        else if (name == "__quantum__qis__cz__body")
                            qc.cz(0, 1);
                        else if (name == "__quantum__qis__mz__body")
                            qc.measure(0, 0);
                        //else if (name == "__quantum__qis__s_adj__body")
                        //    qc.sdg(0);
                        else if (name == "__quantum__qis__s__body")
                            qc.s(0);
                        else if (name == "__quantum__qis__t__body")
                            qc.t(0);
                        else if (name == "__quantum__qis__x__body")
                            qc.x(0);
                        else if (name == "__quantum__qis__y__body")
                            qc.y(0);
                        else if (name == "__quantum__qis__z__body")
                            qc.z(0);
                        else if (name == "__quantum__qis__h__body")
                            qc.h(0);
                        //else if (name == "__quantum__qis__ccx__body")
                        //    qc.ccx(0);
                        else if (name == "__quantum__qis__reset__body")
                            qc.reset(0);
                        //else if (name == "__quantum__qis__swap__body") //TODO Not supported by QMAP?
                        //    qc.swap(0, 1);
                        //else if (name == "__quantum__qis__t_adj__body")
                        //    qc.t(0);
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
