/**
 * @file utilities.hpp
 * @brief
 */

#pragma once

#include <array>
#include <complex>
#include <string>

#define TYPE double

#define HGate "__quantum__qis__h__body"
#define RXGate "__quantum__qis__rx__body"
#define RYGate "__quantum__qis__ry__body"
#define RZGate "__quantum__qis__rz__body"

using Complex = std::complex<TYPE>;

#define I Complex(0, 1)
using ComplexMatrix = std::array<std::array<std::complex<TYPE>, 2>, 2>;
double getTheAngle(Complex theNumber);

ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate);
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate,
                                                    double angle);

// using Complexdouble = std::complex<double>;

// template<typename T>
// using ComplexMatrix2x2 = std::array<std::array<std::complex<T>, 2>, 2>;

// template<typename Type> ComplexMatrix2x2<Type>
// getTheMatrixOfGateFromInstructionName(std::string& theGate);

// template <typename T>
// Complex<T> det(ComplexMatrix2x2<T> mat[2][2]);

Complex det(ComplexMatrix mat);
//ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate);
