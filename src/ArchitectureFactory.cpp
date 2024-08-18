#include "ArchitectureFactory.hpp"
#include <sys-sage.hpp>
#include <cstdint>
#include <set>
#include <utility>

namespace mqt {

Architecture createArchitecture(QDMI_Device dev) {
  
  // QdmiParser: sys-sage's interface to qdmi (for retrieving the static topology)
  QdmiParser qdmi;

  // An instance to QuantumBackend for storing the topology
  QuantumBackend qc = createQcTopo(dev);

  std::uint16_t num_qubits = qc.GetNumberofQubits();

  CouplingMap cm = qc.GetAllCouplingMaps();

  return {num_qubits, cm};  
}
} // namespace mqt
