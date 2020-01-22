from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.tools.monitor import job_monitor
import csv

#Controlled-Controlled -iY Gate, V^2 = -iY
def custom_gate():

    V = 1/np.sqrt(2) * np.array([[1,-1],[1,1]])
    c_v_array = np.kron(np.diag([0,1]),V) + np.diag([1,1,0,0])
    CV = Operator(c_v_array)

    CV_dagger = Operator(c_v_array.conj().T)

    qubits = QuantumRegister(3)
    circuit = QuantumCircuit(qubits)

    circuit.unitary(CV, [qubits[2],qubits[1]])
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(CV_dagger, [qubits[2],qubits[1]], label='c_v_dagger')
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(CV, [qubits[2],qubits[0]], label='c_v')

    return circuit.to_instruction()

def experiment_QC(theta,N,measure_type):

    register_qubits = QuantumRegister(N)
    state_qubit = QuantumRegister(1)
    c = ClassicalRegister(1)
    circ = QuantumCircuit(register_qubits,state_qubit,c)

    #c-iY gate
    U_small = np.kron(np.diag([0,1]),[[0,-1],[1,0]]) + np.diag([1,1,0,0])
    cU = Operator(U_small)


    #Initial Round
    circ.ry(2 * theta,register_qubits[0])
    circ.unitary(cU, [state_qubit[0],register_qubits[0]], label='cU') #target then control
    circ.ry(-2 * theta,register_qubits[0])

    #Subsequent Rounds
    for i in range(1,N):
        circ.cry(np.pi/2,register_qubits[i-1],state_qubit[0]) #reset gate
        circ.ry(2 * theta,register_qubits[i])
        circ.append(custom_gate(),[register_qubits[i-1],register_qubits[i],state_qubit[0]]) #control then target
        circ.ry(-2 * theta,register_qubits[i])

    #Pauli Measurements
    if measure_type == 'x':
        circ.h(state_qubit[0])
    elif measure_type == 'y':
        circ.sdg(state_qubit[0])
        circ.h(state_qubit[0])

    circ.measure(state_qubit[0],c)

    #Simulate Circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend = simulator,shots = 8192)
    result = job.result()
    result_dictionary = result.get_counts(circ)

    probs = {}
    for output in ['0','1']:
        probs[output] = result_dictionary[output]/8192

    return probs['0'] - probs['1']

def computed_fidelity(theta,computed_list):
    q_theta = np.arctan(np.tan(theta)**2)
    expected_list = [np.sin(q_theta),0,np.cos(q_theta)] # <x>,<y>,<z>
    return np.sum([(expected_list[i]-computed_list[i])**2 for i in range(3)])

def analytic_fidelity(theta,n_rounds):
    return 1-(np.cos(theta)**4 + np.sin(theta)**4)**n_rounds

theta = np.pi/4
results = []
for r in range(1,8):
    X_exp = experiment_QC(theta,r,'x')
    Y_exp = experiment_QC(theta,r,'y')
    Z_exp = experiment_QC(theta,r,'z')
    computed_list = [X_exp, Y_exp, Z_exp]
    results.append(1-computed_fidelity(theta,computed_list))

plt.plot(range(1,8),results,label = 'Simulation')
plt.plot(range(1,8),[analytic_fidelity(theta,n) for n in range(1,8)],label = 'Analytic')
plt.legend()
plt.show()
