/**
 * @file utilities.cpp
 * @brief Implementation of common functions.
 *
 */

#include "../headers/utilities.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>

ComplexMatrix getHGate()
{
    /*Matrix Definetation of H Gate*/
    ComplexMatrix Mat;
    Mat[0][0] = 1 / std::sqrt(2);
    Mat[0][1] = 1 / std::sqrt(2);
    Mat[1][0] = 1 / std::sqrt(2);
    Mat[1][1] = -1 / std::sqrt(2);
    return Mat;
}

ComplexMatrix getRXGate(double angle)
{
    /*Matrix Definetation of RX Gate*/
    ComplexMatrix RXate;
    Complex expForm = std::exp(-I * (angle / 2));
    RXate[0][0] = std::cos(angle / 2) * expForm;
    RXate[0][1] = -I * std::sin(angle / 2) * expForm;
    RXate[1][0] = -I * std::sin(angle / 2) * expForm;
    RXate[1][1] = std::cos(angle / 2) * expForm;
    return RXate;
}

Complex det(ComplexMatrix mat)
{
    /*MCalculates determinant of a 2X2 matrix*/
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

double getTheAngle(Complex theNumber)
{
    /* Calculate the angle of a complex number*/
    return std::atan2(std::imag(theNumber), std::real(theNumber));
}

ComplexMatrix getRYGate(double angle)
{
    /*Matrix Definetation of RY Gate*/
    ComplexMatrix RYate;
    Complex expForm = std::exp(-I * (angle / 2));
    RYate[0][0] = std::cos(angle / 2) * expForm;
    RYate[0][1] = -std::sin(angle / 2) * expForm;
    RYate[1][0] = std::sin(angle / 2) * expForm;
    RYate[1][1] = std::cos(angle / 2) * expForm;
    return RYate;
}
ComplexMatrix getRZGate(double angle)
{
    /*Matrix Definetation of RZ Gate*/
    ComplexMatrix RZate;
    Complex expForm = std::exp(-I * (angle / 2));
    RZate[0][0] = std::exp(-I * (angle / 2));
    RZate[0][1] = 0;
    RZate[1][0] = 0;
    RZate[1][1] = std::exp(-I * (angle / 2));
    return RZate;
}
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate)
{
    /*Returns matrix form a gate*/
    ComplexMatrix matrixToReturn = {
        {{1.0, 0.0}, {0.0, 1.0}}}; // Identity Gate for default
    if (theGate == H_Gate)
        matrixToReturn = getHGate();
    return matrixToReturn;
}
ComplexMatrix getTheMatrixOfGateFromInstructionName(std::string theGate,
                                                    double angle = 0)
{
    /*Returns matrix form a gate*/
    ComplexMatrix matrixToReturn = {
        {{1.0, 0.0}, {0.0, 1.0}}}; // Identity Gate for default
    if (theGate == RX_Gate)
        matrixToReturn = getRXGate(angle);
    if (theGate == RY_Gate)
        matrixToReturn = getRYGate(angle);
    if (theGate == RZ_Gate)
        matrixToReturn = getRZGate(angle);
    return matrixToReturn;
}

double mod_2pi(double angle, double atol)
{
    /*Calculates mod2pi of an angle*/
    double wrapped = std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
    if (std::abs(wrapped - M_PI) < atol)
        return -M_PI;
    else
        return wrapped;
}
