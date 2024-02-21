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
    //std::cout << "\n\n0\n\n" << std::endl; 

    auto arch = mqt::createArchitecture(dev);

    // TODO PARSING FROM QIR TO QuantumComputation 
    //      SHOULD HAPPEN HERE
    auto qc = qc::QuantumComputation(arch.getNqubits());
    qc.h(0);
    for (qc::Qubit i = 0; i < arch.getNqubits() - 1; ++i)
        qc.cx(i, i + 1);

    //std::cout << "\n\n3\n\n" << std::endl; 

    auto mapper = HeuristicMapper(qc, arch);
    mapper.map({});
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
