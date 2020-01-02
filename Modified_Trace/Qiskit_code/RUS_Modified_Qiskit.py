from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity,basis_state

def Initial_circuit(theta):
    # Build the initial-sub-circuit
    initial_sub_q = QuantumRegister(2)
    initial_sub_circ = QuantumCircuit(initial_sub_q, name='initial_sub_circ')
    initial_sub_circ.ry(2 * theta,initial_sub_q[0])
    initial_sub_circ.unitary(cU, [initial_sub_q[1],initial_sub_q[0]], label='cU')
    initial_sub_circ.ry(-2 * theta,initial_sub_q[0])
    initial_sub_circ.cry(np.pi / 2,initial_sub_q[0],initial_sub_q[1])
    #print(initial_sub_circ)
    return initial_sub_circ.to_instruction()

def Subsequent_circuit(theta):
    # Build subsequent sub-circuit
    sub_q = QuantumRegister(3) #qubit order: control, ancilla, state
    sub_circ = QuantumCircuit(sub_q, name='sub_circ')
    sub_circ.cry(2 * theta,sub_q[0],sub_q[1])
    sub_circ.unitary(ccU, [sub_q[2],sub_q[1],sub_q[0]], label='ccU') #target then control!
    sub_circ.cry(-2 * theta,sub_q[0],sub_q[1])
    sub_circ.unitary(cReset, [sub_q[2],sub_q[1],sub_q[0]], label='cReset')
    #print(sub_circ)
    return sub_circ.to_instruction()

def desired_state(N):
    return '0' + '1'*(N)


def experiment(theta,N):

    register_qubits = QuantumRegister(N)
    state_qubit = QuantumRegister(1)

    RUS_initial = Initial_circuit(theta)
    RUS = Subsequent_circuit(theta)

    circ = QuantumCircuit(register_qubits,state_qubit)

    circ.append(RUS_initial, [register_qubits[0],state_qubit])

    for i in range(1,N):
        circ.append(RUS, [register_qubits[i-1],register_qubits[i],state_qubit])

    #Simulate Circuit
    backend_sim = BasicAer.get_backend('statevector_simulator')
    result = execute(circ, backend_sim).result()
    state = result.get_statevector(circ)
    #print(state)# state given as qm,qm-1,...q1,q0
    return np.round(state_fidelity(basis_state(desired_state(N), N+1),state),4)

def results(N_max,divisor):

    angles = [np.pi/i for i in divisor]
    for j,k in zip(angles,divisor):
        Success = []
        for i in range(1,N_max+1):
            Success.append(1 - experiment(j,i))
        plt.plot(range(1,i+1),Success,label = '$\\theta = \pi / {}$'.format(k))

    plt.ylabel('$ \ Fidelity, \ \ F(R|\\psi\\rangle,|\\psi\\rangle)$')
    plt.xlabel('$Number \ of \ Ancillae$')
    plt.axhline(1,linestyle = 'dashed',color = 'black')
    plt.legend()
    plt.xlim([1,N_max])
    plt.savefig('Modified_Gearbox_Qiskit.pdf')
    plt.show()

    return Success

#Custom Gates

#Controlled-Controlled -iY Gate
U = np.kron(np.diag([0,0,0,1]),[[0,-1],[1,0]]) + np.diag([1,1,1,1,1,1,0,0])
ccU = Operator(U)

#Controlled -iY Gate
U_small = np.kron(np.diag([0,1]),[[0,-1],[1,0]]) + np.diag([1,1,0,0])
cU = Operator(U_small)

#Controlled-Controlled Ry(pi / 2) Gate
Reset = np.kron(np.diag([0,0,0,1]),(1 / np.sqrt(2)) * np.array([[1,-1],[1,1]]) ) + np.diag([1,1,1,1,1,1,0,0])
cReset = Operator(Reset)

N_max = 10
divisor = [4,6,8]
results(N_max,divisor)
