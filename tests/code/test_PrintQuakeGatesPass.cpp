
// Compile and run with:
// ```
// cudaq-quake test_PrintQuakeGatesPass.cpp -o o.qke  &&
// cudaq-opt --canonicalize --unrolling-pipeline o.qke -o test_PrintQuakeGatesPass.qke
// ```
#include <iostream>
#include <cudaq.h>
#include <fstream>
#include <cudaq/mqss-mqp.h>
// Define a CUDA-Q kernel that is fully specified
// at compile time via templates.
template <std::size_t N>
struct ghz {
  auto operator()() __qpu__ {

    // Compile-time sized array like std::array
    cudaq::qarray<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto kernel = ghz<10>{};
  auto counts = cudaq::sample(kernel);
  counts.dump();
  return 0;
}
