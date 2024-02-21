/**
 * @file QirQmap.cpp
 * @brief Implementation of the 'QirQmapPass' class. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirQmap.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * This pass merges executes qmap to map qubits to architecture.
**/ 

#include <QirQmap.hpp>
#include <iostream>

using namespace llvm;


int convertToQASM(std::string qir, std::string *qasmCode)
{
    Py_Initialize();
    PyObject *pModule = PyImport_ImportModule("hpcqc.qir_qiskit.translate");
    if (!pModule)
    {
        PyErr_Print();
        return 1;
    }
    PyObject *pFunc = PyObject_GetAttrString(pModule, "to_qasm_circuit");
    if (!pFunc)
    {
        PyErr_Print();
        return 2;
    }
    PyObject *pArgs =
    PyTuple_Pack(1, PyBytes_FromStringAndSize(qir.c_str(), qir.size()));
    if (!pArgs)
    {
        PyErr_Print();
        return 3;
    }
    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
    if (!pValue)
    {
        PyErr_Print();
        return 4;
    }
    Py_Finalize();
    *qasmCode = PyUnicode_AsUTF8(pValue);
    return 0;
}

int convertToQIR(std::string qasm, std::string *qir)
{
    Py_Initialize();
    PyObject *pQiskitModule = PyImport_ImportModule("qiskit.QuantumCircuit");
    if (!pQiskitModule)
    {
        PyErr_Print();
        return 1;
    }
    PyObject *pQiskitFunc = PyObject_GetAttrString(pQiskitModule, "from_qasm_str");
    if (!pQiskitFunc)
    {
        PyErr_Print();
        return 2;
    }
    PyObject *pQiskitArgs =
    PyTuple_Pack(1, PyBytes_FromStringAndSize(qasm.c_str(), qasm.size()));
    if (!pQiskitArgs)
    {
        PyErr_Print();
        return 3;
    }
    PyObject *pQiskitValue = PyObject_CallObject(pQiskitFunc, pQiskitArgs);
    if (!pQiskitValue)
    {
        PyErr_Print();
        return 4;
    }
    
    std::string qiskit = PyUnicode_AsUTF8(pQiskitValue);
    
    PyObject *pQIRModule = PyImport_ImportModule("qiskit_qir");
    if (!pQIRModule)
    {
        PyErr_Print();
        return 1;
    }
    PyObject *pQIRFunc = PyObject_GetAttrString(pQIRModule, "to_qir_module");
    if (!pQIRFunc)
    {
        PyErr_Print();
        return 2;
    }
    PyObject *pQIRArgs =
    PyTuple_Pack(1, PyBytes_FromStringAndSize(qiskit.c_str(), qiskit.size()));
    if (!pQIRArgs)
    {
        PyErr_Print();
        return 3;
    }
    PyObject *pQIRValue = PyObject_CallObject(pQIRFunc, pQIRArgs);
    if (!pQIRValue)
    {
        PyErr_Print();
        return 4;
    }

    
    Py_Finalize();
    *qir = PyUnicode_AsUTF8(pQIRValue);
    return 0;
}

/**
 * @brief Applies this pass to the QIR's LLVM module.
 * @param module The module.
 * @param MAM The module analysis manager.
 * @return PreservedAnalyses
 */
PreservedAnalyses QirQmapPass::run(Module &module,
                                             ModuleAnalysisManager & /*MAM*/,
                                             QDMI_Device dev)
{
    auto &Context = module.getContext();
    
    /* TODO:
     * Transform QIR to Qiskit
     * Get arch if you need
     *  qmap
     * Transform qmap result to qiskit
     * Transform qiskit to qir
     */
    std::string* result1;
    std::string* result2;

    //convertToQASM("", result1);
    convertToQIR("", result2);
   
    return PreservedAnalyses::none();
}

/**
 * @brief External function for loading the 'QirQmapPass' as a
 * 'PassModule'.
 * @return QirQmapPass
 */
extern "C" PassModule *loadQirPass() { return new QirQmapPass(); }
