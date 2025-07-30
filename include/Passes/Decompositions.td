include "mlir/Pass/PassBase.td"

def CxToHCzH : Pass<"CxToHCzH"> {
  let summary = "Decomposition pass of two-qubits cnot by H, Cz, and H";
  let constructor = "mqss::opt::createCxToHCzHDecompositionPass()";
}

def CzToHCxH : Pass<"CzToHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToHCxHDecompositionPass()";
}

def ReverseCx : Pass<"ReverseCx"> {
  let summary = "Decomposition pass that reverses the control and targets of each two-qubits CNot gate in a circuit";
  let constructor = "mqss::opt::createReverseCxPass()";
}

def SAdjZToS : Pass<"SAdjZToS"> {
  let summary = "Optimization pass that replaces a pattern composed of S adjoint and Z by S";
  let constructor = "mqss::opt::createSAdjZToSPass()";
}

def SZToSAdj : Pass<"SZToSAdj"> {
  let summary = " Optimization pass that replaces a pattern composed of S and Z by S(adj)";
  let constructor = "mqss::opt::createSZToSAdjPass()";
}
