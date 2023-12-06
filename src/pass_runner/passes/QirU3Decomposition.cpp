/**
 * @file QirU3DecompositionPass.cpp
 * @brief Implementation of the 'QirU3DecompositionPass' class.
 * This Pass creates a U3 Decomposition of given gates (gatesToDecompose).
 * It uses getDecompositionAngles function of QirZYZDecompositionPass to get
 * angles. We ignore Phase since it is not supported by QIR (As December 2023)
 * Example, H q[0] -> U3(a, b, c) q[0]
 * <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirU3Decomposition.cpp?ref_type=heads">
 * Go to the source code of this file.</a>
 *
 */

#include "../headers/QirU3Decomposition.hpp"
#include "../headers/QirZYZDecomposition.hpp"
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace llvm;

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */

PreservedAnalyses QirU3DecompositionPass::run(Module &module,
                                              ModuleAnalysisManager & /*MAM*/)
{
    std::string gatesToDecompose[4] = {
        "__quantum__qis__rx__body", "__quantum__qis__ry__body",
        "__quantum__qis__rz__body", "__quantum__qis__h__body"};

    std::vector<Instruction *> gatesToErase;
    FunctionCallee U3 = nullptr;
    LLVMContext &rContext = module.getContext();
    IRBuilder<> builder(rContext);
    QirZYZDecompositionPass ZYZPass;
    for (auto &function : module)
    {
        for (auto &block : function)
        {
            for (auto &instruction : block)
            {
                CallInst *callInstr = dyn_cast<CallInst>(&instruction);
                if (!callInstr)
                    continue;
                Function *calledFunction = callInstr->getCalledFunction();
                for (std::string gateToDecompose : gatesToDecompose)
                {
                    if (gateToDecompose == calledFunction->getName())
                    {
                        int numberOfOperand = callInstr->getNumOperands();
                        Value *theLastOperand =
                            callInstr->getOperand(numberOfOperand - 2);
                        Value *theAngle = callInstr->getOperand(0);
                        LoadInst *loadofTheAngle = dyn_cast<LoadInst>(theAngle);
                        ComplexMatrix theGate;
                        if (!loadofTheAngle)
                        {
                            theGate = getTheMatrixOfGateFromInstructionName(
                                gateToDecompose);
                        }
                        else
                        {
                            Value *theRotationAngle =
                                loadofTheAngle->getPointerOperand();
                            GlobalVariable *angleAsAConst =
                                dyn_cast_or_null<GlobalVariable>(
                                    theRotationAngle);
                            ConstantFP *angleFP = dyn_cast_or_null<ConstantFP>(
                                angleAsAConst->getInitializer());
                            double angle =
                                angleFP->getValue().convertToDouble();
                            theGate = getTheMatrixOfGateFromInstructionName(
                                gateToDecompose, angle);
                        }
                        if (!U3)
                        {
                            Type *qubitType = theLastOperand->getType();
                            FunctionType *rotationGateType = FunctionType::get(
                                Type::getVoidTy(rContext),
                                {Type::getDoubleTy(rContext),
                                 Type::getDoubleTy(rContext),
                                 Type::getDoubleTy(rContext), qubitType},
                                false);

                            errs() << *rotationGateType << "\n";
                            U3 = module.getOrInsertFunction(U3_Gate,
                                                            rotationGateType);
                        }

                        builder.SetInsertPoint((&instruction));
                        std::vector<Value *> theAngles =
                            ZYZPass.getDecompositionAngles(rContext, theGate);
                        std::vector<Value *> args = {theAngles[1], theAngles[0],
                                                     theAngles[2],
                                                     theLastOperand};
                        builder.CreateCall(U3, args);
                        gatesToErase.push_back(&instruction);
                    }
                }
            }
        }
    }

    for (Instruction *instr : gatesToErase)
    {
        instr->eraseFromParent();
    }
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirU3DecompositionPass' as a
 * 'PassModule'.
 * @return QirU3DecompositionPass
 */
extern "C" PassModule *loadQirPass() { return new QirU3DecompositionPass(); }
