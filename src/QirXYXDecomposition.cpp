/**
 * @file QirXYXDecompositionPass.cpp
 * @brief Implementation of the 'QirXYXDecompositionPass' class.
 * This Pass creates a XYX Decomposition of given gates (gatesToDecompose).
 * It uses getDecompositionAnglesAsNumber function of QirZYZDecompositionPass to
 * get angles. We ignore Phase since it is not supported by QIR (As December
 * 2023) Example, H q[0] -> RX(a) q[0]; RY(b) q[0]; RX(c) q[0]; <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirXYXDecomposition.cpp?ref_type=heads">
 * Go to the source code of this file.</a>
 *
 */

#include <QirXYXDecomposition.hpp>
#include <QirZYZDecomposition.hpp>
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

PreservedAnalyses QirXYXDecompositionPass::run(Module &module,
                                               ModuleAnalysisManager & /*MAM*/)
{
    std::string gatesToDecompose[4] = {
        "__quantum__qis__rx__body", "__quantum__qis__ry__body",
        "__quantum__qis__rz__body", "__quantum__qis__h__body"};

    std::vector<Instruction *> gatesToErase;
    FunctionCallee RX = nullptr;
    FunctionCallee RY = nullptr;
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
                        if (!RX)
                        {
                            Type *qubitType = theLastOperand->getType();
                            FunctionType *rotationGateType = FunctionType::get(
                                Type::getVoidTy(rContext),
                                {Type::getDoubleTy(rContext), qubitType},
                                false);
                            RX = module.getOrInsertFunction(RX_Gate,
                                                            rotationGateType);
                            RY = module.getOrInsertFunction(RY_Gate,
                                                            rotationGateType);
                        }

                        builder.SetInsertPoint((&instruction));
                        std::vector<double> theAngles =
                            ZYZPass.getDecompositionAnglesAsNumber(rContext,
                                                                   theGate);
                        double phi = theAngles[0];
                        double theta = theAngles[1];
                        double lam = theAngles[2];
                        double new_phi = mod_2pi(phi + M_PI, 0.);
                        double new_lam = mod_2pi(lam + M_PI, 0.);
                        Value *phiValue = ConstantFP::get(
                            rContext, APFloat(static_cast<float>(new_phi)));
                        Value *lamValue = ConstantFP::get(
                            rContext, APFloat(static_cast<float>(new_lam)));
                        Value *thetaValue = ConstantFP::get(
                            rContext, APFloat(static_cast<float>(theta)));
                        builder.CreateCall(RX, {phiValue, theLastOperand});
                        builder.CreateCall(RY, {thetaValue, theLastOperand});
                        builder.CreateCall(RX, {lamValue, theLastOperand});
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
 * @brief External function for loading the 'QirXYXDecompositionPass' as a
 * 'PassModule'.
 * @return QirXYXDecompositionPass
 */
extern "C" AgnosticPassModule *loadQirPass()
{
    return new QirXYXDecompositionPass();
}
