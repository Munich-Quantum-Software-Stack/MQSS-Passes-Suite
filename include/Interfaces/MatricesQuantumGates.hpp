#pragma once

#include <Eigen/Dense>
#include <cmath> // for std::cos, std::sin, std::sqrt, M_PI
#include <complex>

using namespace std;
using namespace Eigen;

class QuantumGates {
public:
  // Single-qubit gates (2x2 matrices)
  static Matrix2cd X() {
    Matrix2cd m;
    m << 0, 1, 1, 0;
    return m;
  }

  static Matrix2cd Y() {
    Matrix2cd m;
    m << 0, -std::complex<double>(0, 1), std::complex<double>(0, 1), 0;
    return m;
  }

  static Matrix2cd Z() {
    Matrix2cd m;
    m << 1, 0, 0, -1;
    return m;
  }

  static Matrix2cd H() {
    Matrix2cd m;
    m << 1, 1, 1, -1;
    return (1.0 / std::sqrt(2)) * m;
  }

  static Matrix2cd S() {
    Matrix2cd m = Matrix2cd::Identity();
    m(1, 1) = std::exp(std::complex<double>(0, M_PI / 2));
    return m;
  }

  static Matrix2cd T() {
    Matrix2cd m = Matrix2cd::Identity();
    m(1, 1) = std::exp(std::complex<double>(0, M_PI / 4));
    return m;
  }

  // Rotation R1 (phase rotation) gate
  static Matrix2cd R1(double theta) {
    Matrix2cd m = Matrix2cd::Identity();
    m(1, 1) = std::exp(std::complex<double>(0, theta));
    return m;
  }

  // U3 gate with parameters theta, phi, lambda
  static Matrix2cd U3(double theta, double phi, double lambda) {
    Matrix2cd m;
    using std::cos;
    using std::exp;
    using std::sin;

    m(0, 0) = cos(theta / 2);
    m(0, 1) = -exp(std::complex<double>(0, lambda)) * sin(theta / 2);
    m(1, 0) = exp(std::complex<double>(0, phi)) * sin(theta / 2);
    m(1, 1) = exp(std::complex<double>(0, phi + lambda)) * cos(theta / 2);

    return m;
  }

  // Two-qubit gates (4x4 matrices)

  static Matrix4cd CNOT() {
    Matrix4cd m = Matrix4cd::Zero();
    m(0, 0) = 1;
    m(1, 1) = 1;
    m(2, 3) = 1;
    m(3, 2) = 1;
    return m;
  }

  static Matrix4cd CZ() {
    Matrix4cd m = Matrix4cd::Identity();
    m(3, 3) = -1;
    return m;
  }

  static Matrix4cd CY() {
    Matrix4cd m = Matrix4cd::Zero();
    m(0, 0) = 1;
    m(1, 1) = 1;
    m(2, 3) = -std::complex<double>(0, 1);
    m(3, 2) = std::complex<double>(0, 1);
    return m;
  }

  static Matrix4cd SWAP() {
    Matrix4cd m = Matrix4cd::Zero();
    m(0, 0) = 1;
    m(1, 2) = 1;
    m(2, 1) = 1;
    m(3, 3) = 1;
    return m;
  }

  static Matrix<std::complex<double>, 8, 8> CSWAP() {
    Matrix<std::complex<double>, 8, 8> m =
        Matrix<std::complex<double>, 8, 8>::Identity();
    // Swap the last two qubits conditioned on the first qubit
    m(5, 6) = 1;
    m(6, 5) = 1;
    m(5, 5) = 0;
    m(6, 6) = 0;
    return m;
  }

  static Matrix2cd Rx(double theta) {
    using std::cos;
    using std::sin;

    Matrix2cd m;
    m << cos(theta / 2), -1.i * sin(theta / 2), -1.i * sin(theta / 2),
        cos(theta / 2);
    return m;
  }

  static Matrix2cd Ry(double theta) {
    Matrix2cd m;
    using std::cos;
    using std::sin;
    m(0, 0) = cos(theta / 2);
    m(0, 1) = -sin(theta / 2);
    m(1, 0) = sin(theta / 2);
    m(1, 1) = cos(theta / 2);
    return m;
  }
  static Matrix2cd Rz(double theta) {
    Matrix2cd m = Matrix2cd::Zero();
    m(0, 0) = std::exp(std::complex<double>(0, -theta / 2));
    m(1, 1) = std::exp(std::complex<double>(0, theta / 2));
    return m;
  }
};
