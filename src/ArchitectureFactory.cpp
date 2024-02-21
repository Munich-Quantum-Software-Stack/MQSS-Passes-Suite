#include "ArchitectureFactory.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace mqt {

Architecture createArchitecture(QDMI_Device dev) {
  int num_qubits = 0;
  //auto err = QDMI_query_device_property_exists(dev, QDMI_QUBIT_COUNT, nullptr);
  int err = QDMI_query_qubits_num(dev, &num_qubits);
  if (err != QDMI_SUCCESS || num_qubits == 0) 
    throw std::runtime_error("Could not get number of qubits via QDMI");

  QDMI_Qubit qubits;

  // create a dummy coupling map 
  err = QDMI_query_all_qubits(dev, &qubits);

  if (err != QDMI_SUCCESS || qubits == NULL)
    throw std::runtime_error("Could not get qubits via QDMI");

  CouplingMap cm{};
  for (int i = 0; i < num_qubits; i++)
  {
    if (qubits[i].coupling_mapping == NULL)
      throw std::runtime_error("Could not get number of qubits via QDMI");

    for (int j = 0; j < qubits[i].size_coupling_mapping; j++)
      cm.emplace(i, qubits[i].coupling_mapping[j]);
  }

  free(qubits);

  return {static_cast<std::uint16_t>(num_qubits), cm};
}
} // namespace mqt
