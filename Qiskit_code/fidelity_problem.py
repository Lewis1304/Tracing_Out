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
    circ.unitary(cU, [state_qubit,register_qubits[0]], label='cU') #target then control
    circ.ry(-2 * theta,register_qubits[0])
    circ.cry(np.pi/2,register_qubits[0],*state_qubit) #reset gate

    #Subsequent Rounds
    for i in range(1,N):
        circ.ry(2 * theta,register_qubits[i])
        circ.append(custom_gate(),[register_qubits[i-1],register_qubits[i],state_qubit]) #control then target
        circ.ry(-2 * theta,register_qubits[i])
        circ.cry(np.pi/2,register_qubits[i],*state_qubit) #reset gate


    #Pauli Measurements
    if measure_type == 'x':
        circ.h(state_qubit)
    elif measure_type == 'y':
        circ.sdg(state_qubit)
        circ.h(state_qubit)
    elif measure_type == 'z':
        circ.iden(state_qubit)
    circ.measure(state_qubit,c)

    #Simulate Circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend = simulator,shots = 15000)
    result = job.result()
    result_dictionary = result.get_counts(circ)

    probs = {}
    for output in ['0','1']:
        if output in result_dictionary:
            probs[output] = result_dictionary[output]
        else:
            probs[output] = 0
    return (probs['0'] -  probs['1']) / 15000

def testing_activation_function():
    x = np.linspace(0,np.pi/2,25)
    for r in range(1,7):
        results = []
        for i in x:
            X_exp = experiment_QC(i,r,'x')
            Y_exp = experiment_QC(i,r,'y')
            Z_exp = experiment_QC(i,r,'z')
            print(X_exp**2 + Y_exp**2 + Z_exp**2)
            results.append(np.arctan2(X_exp,Z_exp)/2)
        plt.plot(x,results,label = 'Round: {}'.format(r))
        print('{} / 15'.format(r))
    plt.legend()
    plt.show()


def q(theta):
    return np.arctan(np.tan(theta)**2)

def p(theta):
    return np.cos(theta)**4+ np.sin(theta)**4

def analytic_evs(theta, N):
    """analytic expectation values
    """
    return (1-(1-p(theta))**N)*np.array([2*np.cos(theta)**2*np.sin(theta)**2, 0, np.cos(theta)**4-np.sin(theta)**4])/p(theta)+(1-p(theta))**N*np.array([0, 0, 1])

def testing_expectation_values():
    x = np.linspace(0,np.pi/2,25)
    for r in range(1,4):
        results = []
        for i in x:
            X_exp = experiment_QC(i,r,'x')
            Y_exp = experiment_QC(i,r,'y')
            Z_exp = experiment_QC(i,r,'z')
            results.append([X_exp, Y_exp, Z_exp])
        plt.plot(x,np.array(results),label = 'Round: {}'.format(r))
        plt.plot(x, np.array([analytic_evs(i, r) for i in x]), linestyle='dashed', label='analytic')
        print('{} / 15'.format(r))
    plt.legend()
    plt.show()

def computed_fidelity(theta,X_exp,Y_exp,Z_exp):
    computed_list = np.array([X_exp,Y_exp,Z_exp])
    expected_list = np.array([2*np.cos(theta)**2*np.sin(theta)**2, 0, np.cos(theta)**4-np.sin(theta)**4])/p(theta)
    return 0.5*(1+np.dot(computed_list, expected_list))

def analytic_fidelity(theta,n_rounds):
    return (1 - (1-p(theta))**n_rounds*(1-np.cos(theta)**4/p(theta)))

def testing_fidelity():
    theta = np.pi/4
    x = range(1,10)
    results = []
    for i in x:
        print('Round {} / 15'.format(i))
        X_exp = experiment_QC(theta,i,'x')
        Y_exp = experiment_QC(theta,i,'y')
        Z_exp = experiment_QC(theta,i,'z')

        comp = computed_fidelity(theta,X_exp,Y_exp,Z_exp)
        an = analytic_fidelity(theta, i)
        results.append(comp)

    plt.plot(x,results,label = 'Simulation')
    plt.plot(x,[analytic_fidelity(theta,n) for n in x], linestyle='dashed', label = 'Analytic')
    plt.legend()
    plt.show()

testing_fidelity()
#testing_expectation_values()
#x = np.linspace(0, np.pi/2, 25)
#plt.plot([np.arctan2(e[0], e[2]) for e in analytic_evss])
#plt.show()
