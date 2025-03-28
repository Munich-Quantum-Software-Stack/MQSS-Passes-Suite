include "mlir/Pass/PassBase.td"

def CxToHCzHDecompositionPass: Pass<"CxToHCzH"> {
  let summary = "Decomposition pass of two-qubits cnot by H, Cz, and H";
  let constructor = "mqss::opt::createCxToHCzHDecompositionPass()";
}

def CzToHCxHDecompositionPass: Pass<"CzToHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToHCxHDecompositionPass()";
}

def ReverseCNotPass: Pass<"ReverseCx"> {
  let summary = "Decomposition pass that reverses the control and targets of each two-qubits CNot gate in a circuit";
  let constructor = "mqss::opt::createReverseCNotPass()";
}

def SAdjToSPass : Pass<"SadjZToS"> {
  let summary = "Optimization pass that replaces a pattern composed of S adjoint and Z by S";
  let constructor = "mqss::opt::createSAdjToSPass()";
}

def SToSAdjPass: Pass<"SZToSadj"> {
  let summary = " Optimization pass that replaces a pattern composed of S and Z by S(adj)";
  let constructor = "mqss::opt::createSToSAdjPass()";
}
