/**
 * @file QirAllocationAnalysis.cpp
 * @brief Implementation of the 'QirAllocationAnalysisPass' analysis pass. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirAllocationAnalysis.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/StaticResourceComponent/AllocationAnalysisPass.cpp
 */

#include "PassModule.hpp"
#include <QirPythonAPITest.hpp>

using namespace llvm;

/**
 * @brief Applies an analysis pass to the 'function' function.
 * @param function The function.
 * @param FAM The function analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirPythonAPITest::run(Module &module,
                                        ModuleAnalysisManager &MAM,
                                        QDMI_Device dev)
{
    std::string pythonStr = "It is really good string";
    Py_Initialize();
    PyObject *pModule = PyImport_ImportModule("Some Python Module");
    if (!pModule)
    {
        PyErr_Print();
    }
    PyObject *pFunc = PyObject_GetAttrString(pModule, "Some Python Function");
    if (!pFunc)
    {
        PyErr_Print();
    }
    PyObject *pArgs =
        PyTuple_Pack(1, PyBytes_FromStringAndSize(pythonStr.c_str(), pythonStr.size()));
    if (!pArgs)
    {
        PyErr_Print();
    }
    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
    // It is the return value of the function
    if (!pValue)
    {
        PyErr_Print();
    }
    Py_Finalize();

    return PreservedAnalyses::none();
}

extern "C" PassModule *loadQirPass() { return new QirPythonAPITest(); }