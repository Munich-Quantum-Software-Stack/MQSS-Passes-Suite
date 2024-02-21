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

    // TODO PARSING FROM LLVM::Module TO QuantumComputation 
    //      SHOULD HAPPEN HERE
    auto qc = qc::QuantumComputation(arch.getNqubits());
    qc.h(0);
    for (qc::Qubit i = 0; i < arch.getNqubits() - 1; ++i)
        qc.cx(i, i + 1);

    auto mapper = HeuristicMapper(qc, arch);
    mapper.map({});
    mapper.dumpResult(std::cout, qc::Format::OpenQASM3);

    // TODO PARSING FROM QASM3 TO LLVM::Module SHOULD HAPPEN 
    //      HERE

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
