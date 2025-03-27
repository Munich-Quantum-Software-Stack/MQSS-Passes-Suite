include "mlir/Pass/PassBase.td"

def CxToHCzHDecompositionPass: Pass<"CxToHCzH"> {
  let summary = "Decomposition pass of two-qubits cnot by H, Cz, and H";
  let constructor = "createCxToHCzHDecompositionPass()";
}

def CzToHCxHDecompositionPass: Pass<"CzToHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "createCzToHCxHDecompositionPass()";
}

def ReverseCNotPass: Pass<"ReverseCx"> {
  let summary = "Decomposition pass that reverses the control and targets of each two-qubits CNot gate in a circuit";
  let constructor = "createReverseCNotPass()";
}

def SAdjToSPass : Pass<"SadjZToS"> {
  let summary = "Optimization pass that replaces a pattern composed of S adjoint and Z by S";
  let constructor = "createSAdjToSPass()";
}

def SToSAdjPass: Pass<"SZToSadj"> {
  let summary = " Optimization pass that replaces a pattern composed of S and Z by S(adj)";
  let constructor = "createSToSAdjPass()";
}
