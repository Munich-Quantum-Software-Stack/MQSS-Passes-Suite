#include "ArchitectureFactory.hpp"
#include <sys-sage.hpp>
#include <cstdint>
#include <set>
#include <utility>

namespace mqt {

Architecture createArchitecture(QDMI_Device dev) {
  
  // QDMI_Parser: sys-sage's interface to qdmi (for retrieving the static topology)
  QDMI_Parser qdmi;

  // An instance to QuantumBackend for storing the topology
  QuantumBackend* qc = new QuantumBackend(0, "IBM_Backend");

  // Create the topology
  qdmi.createQcTopo(qc, dev);

  std::uint16_t num_qubits = qc->GetNumberofQubits();

  CouplingMap cm = qc->GetAllCouplingMaps();

  delete qc;
  return {num_qubits, cm};  
}
} // namespace mqt
