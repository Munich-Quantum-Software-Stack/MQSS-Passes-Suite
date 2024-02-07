/**
 * @file utilities.hpp
 * @brief
 */

#pragma once

#include <array>
#include <complex>
#include <string>

#define TYPE double

#define H_Gate "__quantum__qis__h__body"
#define RX_Gate "__quantum__qis__rx__body"
#define RY_Gate "__quantum__qis__ry__body"
#define RZ_Gate "__quantum__qis__rz__body"
#define U3_Gate "__quantum__qis__U3__body"

using Complex = std::complex<TYPE>;

//#define I Complex(0, 1)
constexpr Complex I = Complex(0, 1);
using ComplexMatrix = std::array<std::array<std::complex<TYPE>, 2>, 2>;
double getTheAngle(Complex theNumber);

ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate);
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate,
                                                    double angle);

double mod_2pi(double angle, double atol);

Complex det(ComplexMatrix mat);
