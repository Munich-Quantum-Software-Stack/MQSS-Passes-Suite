// Compile and run with:
// ```
// cudaq-quake QuakeToTikzPass.cpp -o o.qke  &&
// cudaq-opt --canonicalize --unrolling-pipeline o.qke -o QuakeToTikzPass.qke
// ```

#include <iostream>
#include <cudaq.h>
#include <fstream>

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
    swap(q[0],q[3]);
    rx(2.34,q[0]);
    ry(3.5,q[1]);
    rz(5.0,q[2]);
    x<cudaq::ctrl>(q[0], q[1],q[3]);
    mz(q);

  }
};

int main() {
  auto kernel = ghz<6>{};
  auto counts = cudaq::sample(kernel);
  counts.dump();
  return 0;
}
