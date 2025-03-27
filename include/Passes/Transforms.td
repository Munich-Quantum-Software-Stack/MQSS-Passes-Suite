include "mlir/Pass/PassBase.td"

def CommuteCNotRxPass : Pass<"CommuteCxRx"> {
  let summary = "Apply commutation pass of pattern CNot-Rx";
  let constructor = "createCommuteCNotRxPass()";
}

def  CommuteCNotXPass: Pass<"CommuteCxX"> {
  let summary = "Apply commutation pass to pattern CNot-X";
  let constructor = "createCommuteCNotXPass()";
}

def CommuteCNotZPass: Pass<"CommuteCxZ"> {
  let summary = "Apply commutation pass to pattern CNot-Z to Z-CNot";
  let constructor = "createCommuteCNotZPass()";
}

def CommuteRxCNotPass: Pass<"CommuteRxCx"> {
  let summary = "Apply commutation pass to pattern Rx-CNot to CNot-Rx";
  let constructor = "createCommuteRxCNotPass()";
}

def CommuteXCNotPass: Pass<"CommuteXCx"> {
  let summary = "Apply commutation pass to pattern X-CNot to CNot-X";
  let constructor = "createCommuteXCNotPass()";
}

def CommuteZCNotPass: Pass<"CommuteZCx"> {
  let summary = "Apply commutation pass to pattern Z-CNot to CNot-Z";
  let constructor = "createCommuteZCNotPass()";
}

def DoubleCNotCancellationPass : Pass<"CancellationDoubleCx"> {
  let summary = "This pass removes the pattern CNot, CNot if both gates operates on the same control and targets. ";
  let constructor = "createDoubleCnotCancellationPass()";
}

def HXHToZPass: Pass<"HXHToZ"> {
  let summary = " Optimization pass that replaces a pattern composed of H, X, H by Z";
  let constructor = "createHXHToZPass()";
}

def HZHToXPass : Pass<"HZHToX"> {
  let summary = "Optimization pass that replaces a pattern composed of H, Z, H by X";
  let constructor = "createHZHToXPass()";
}

def HadamardAndXGateSwitchPass: Pass<"SwitchHX"> {
  let summary = "Pass that switches a pattern composed Hadamard and X to Z and Hadamard";
  let constructor = "createHadamardAndXGateSwitchPass()";
}

def HadamardAndYGateSwitchPass: Pass<"SwitchHY"> {
  let summary = "Pass that switches a pattern composed Hadamard and Y to Y and Hadamard";
  let constructor = "createHadamardAndYGateSwitchPass()";
}

def HadamardAndZGateSwitchPass: Pass<"SwitchHZ"> {
  let summary = "Pass that switches a pattern composed Hadamard and Z to X and Hadamard";
  let constructor = "createHadamardAndZGateSwitchPass()";
}

def NormalizeArgAnglePass: Pass<"NormalizeArgAngle"> {
  let summary = "Optimization pass that normalizes the angle of Rx, Ry and Rz rotations";
  let constructor = "createNormalizeArgAnglePass()";
}

def NullRotationCancellationPass: Pass<"CancelNullRotations"> {
  let summary = "Optimization pass that removes of Rx, Ry and Rz null rotations";
  let constructor = " createNullRotationCancellationPass()";
}

def PauliGateAndHadamardSwitchPass: Pass<"SwitchPauliH"> {
  let summary = "Pass that switches a pattern composed by {X,Y,Z} (Pauli) and Hadamard";
  let constructor = "createPauliGateAndHadamardSwitchPass()";
}

def XGateAndHadamardSwitchPass: Pass<"SwitchXH"> {
  let summary = "Pass that switches a pattern composed by X and Hadamard to Hadamard and Z";
  let constructor = "createXGateAndHadamardSwitchPass()";
}

def YGateAndHadamardSwitchPass: Pass<"SwitchYH"> {
  let summary = "Pass that switches a pattern composed by Y and Hadamard to Hadamard and Y";
  let constructor = "createYGateAndHadamardSwitchPass()";
}

def ZGateAndHadamardSwitchPass: Pass<"SwitchZH"> {
  let summary = "Pass that switches a pattern composed by Z and Hadamard to Hadamard and X";
  let constructor = "createZGateAndHadamardSwitchPass()";
}
