#include "ArchitectureFactory.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace mqt {

Architecture createArchitecture(QDMI_Device dev) {
  auto err = QDMI_query_device_property_exists(dev, QDMI_QUBIT_COUNT, nullptr);
  if (QDMI_IS_ERROR(err)) {
    throw std::runtime_error("Could not get number of qubits via QDMI");
  }
  int numQubits{};
  err = QDMI_query_device_property_i(dev, QDMI_QUBIT_COUNT, &numQubits);
  if (QDMI_IS_ERROR(err)) {
    throw std::runtime_error("Could not get number of qubits via QDMI");
  }

  // create a dummy coupling map for now using a line topology
  CouplingMap cm{};
  for (int i = 0; i < numQubits - 1; ++i) {
    cm.emplace(i, i + 1);
  }
  return {static_cast<std::uint16_t>(numQubits), cm};
}
} // namespace mqt
