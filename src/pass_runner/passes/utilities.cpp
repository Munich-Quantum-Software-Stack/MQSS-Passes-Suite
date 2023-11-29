/**
 * @file QirAllocationAnalysis.cpp
 * @brief Implementation of the 'QirAllocationAnalysisPass' analysis pass. <a
 * href="https://gitlab-int.srv.lrz.de/lrz-qct-qis/quantum_intermediate_representation/qir_passes/-/blob/Plugins/src/passes/QirAllocationAnalysis.cpp?ref_type=heads">Go
 * to the source code of this file.</a>
 *
 * Adapted from:
 * https://github.com/qir-alliance/qat/blob/main/qir/qat/Passes/StaticResourceComponent/AllocationAnalysisPass.cpp
 */

#include "../headers/utilities.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>

ComplexMatrix getHGate() {
  ComplexMatrix Mat;
  Mat[0][0] = 1 / std::sqrt(2);
  Mat[0][1] = 1 / std::sqrt(2);
  Mat[1][0] = 1 / std::sqrt(2);
  Mat[1][1] = -1 / std::sqrt(2);
  return Mat;
}

ComplexMatrix getRXGate(double angle) {
  ComplexMatrix RXate;
  Complex expForm = std::exp(-I * (angle / 2));
  RXate[0][0] = std::cos(angle / 2) * expForm;
  RXate[0][1] = -I * std::sin(angle / 2) * expForm;
  RXate[1][0] = -I * std::sin(angle / 2) * expForm;
  RXate[1][1] = std::cos(angle / 2) * expForm;
  return RXate;
}

Complex det(ComplexMatrix mat) {
  return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

double getTheAngle(Complex theNumber) {
  return std::atan2(std::imag(theNumber), std::real(theNumber));
}

ComplexMatrix getRYGate(double angle) {
  ComplexMatrix RYate;
  Complex expForm = std::exp(-I * (angle / 2));
  RYate[0][0] = std::cos(angle / 2) * expForm;
  RYate[0][1] = -std::sin(angle / 2) * expForm;
  RYate[1][0] = std::sin(angle / 2) * expForm;
  RYate[1][1] = std::cos(angle / 2) * expForm;
  return RYate;
}
ComplexMatrix getRZGate(double angle) {
  ComplexMatrix RZate;
  Complex expForm = std::exp(-I * (angle / 2));
  RZate[0][0] = std::exp(-I * (angle / 2));
  RZate[0][1] = 0;
  RZate[1][0] = 0;
  RZate[1][1] = std::exp(-I * (angle / 2));
  return RZate;
}
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate) {
  if (theGate == HGate)
    return getHGate();
  
  ComplexMatrix identityGate = {{{1.0, 0.0}, {0.0, 1.0}}};
  return identityGate;
}
/*
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate,
                                                    double angle) {
  if (theGate == RXGate)
    return getRXGate(angle);
}

*/
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate,
                                                    double angle = 0) {
  if (theGate == RXGate)
    return getRXGate(angle);
  
  ComplexMatrix identityGate = {{{1.0, 0.0}, {0.0, 1.0}}};
  return identityGate;
}
