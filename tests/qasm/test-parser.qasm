// Benchmark was created by MQT Bench on 2024-03-17
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
qreg q2[3];
creg meas[13];
cx q2[0],q2[1];
cx q2[1],q2[2];
h q2[2];
cx q2[0],q[1];
h q[9];
id q[9];
rx(3.21) q[0];
ry(3.1416) q[0];
rz(-3.1416) q[9];
ccx q[1],q[2],q[3];
cx q[9],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
CX q[5],q[4];
CX q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
cy q[2],q[1];
cz q[1],q[0];
//sx q[0]; check equivalence
//sxdg q[0]; check equivalence
//phase (1.5) q[3]; check equivalence
//p (1.3) q2[1]; check equivalence
t q[9];
s q[3];
s q[8];
x q[1];
x q[3];
y q[4];
y q[3];
z q[8];
z q[3];
u1(1.4) q[1];
u3(1.4,3.2,1.4) q[1];
//u2(1.4,2.3) q[1]; check equivalence
//u(1.4,1.3,1.0) q[1]; check equivalence
swap q[4],q[3];
//iswap q[4],q[3]; chech equivalence
//iswapdg q[4],q[3]; chech equivalence
ch q[0], q[1];
ch q[9], q2[1];
ch q2[0], q2[1];
crx(3.21) q[0], q[4];
cry(3.1416) q[6], q[7];
//rxx(0.5767575625234357) q[1],q[4]; chech equivalence
//ryy(0.5767575625234357) q[1],q[4]; chech equivalence
//rzz(0.5767575625234357) q[1],q[4]; chech equivalence
//rzx(0.5767575625234357) q[1],q[4]; chech equivalence
dcx q2[0], q2[1];
//ecr q2[0], q2[1];
//cswap q[0], q[1], q[3]; chech equivalence
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q2[0],q2[1],q2[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q2[0] -> meas[10];
measure q2[1] -> meas[11];
measure q2[2] -> meas[12];
