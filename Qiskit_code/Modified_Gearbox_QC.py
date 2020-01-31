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

def experiment_QC(theta,N,name,measure_type):

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

    if name == 'qasm_simulator':
        shots = 50000
        simulator = Aer.get_backend(name)
        job = execute(circ, backend = simulator,shots = shots)
        result = job.result()
        result_dictionary = result.get_counts(circ)

        probs = {}
        for output in ['0','1']:
            if output in result_dictionary:
                probs[output] = result_dictionary[output]
            else:
                probs[output] = 0

        return (probs['0'] -  probs['1']) / shots

    else:
        shots = 8192
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend(name)

        #stats about the circuit implemented on the actual Quantum Computer- Different to theoretical circuit!:
        #t_circ = transpile(circ, backend=qcomp)
        #print('Circuit Depth: ',t_circ.depth())
        #print('Circuit Width: ',t_circ.width())
        #print('Total Number of Operations: ',t_circ.size())
        #print('Amount of paralell subcircuits: ',t_circ.num_tensor_factors())

        job = execute(circ, backend = qcomp,shots = shots)
        print(job_monitor(job))
        result = job.result()
        result_dictionary = result.get_counts(circ)

        probs = {}
        for output in ['0','1']:
            if output in result_dictionary:
                probs[output] = result_dictionary[output]
            else:
                probs[output] = 0
        return (probs['0'] -  probs['1']) /shots


###############################################################################
#Gathering Data

acquiring_simulation_data = False
if acquiring_simulation_data == True:
    x = np.linspace(0,np.pi/2,25)
    with open('data_sim.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Theta', 'Rounds', '<X>', '<Y>', '<Z>'])

        for step,theta in enumerate(x):
            for n_rounds in range(1,9):
                    N = n_rounds
                    X_exp = experiment_QC(theta,N,'qasm_simulator','x')
                    Y_exp = experiment_QC(theta,N,'qasm_simulator','y')
                    Z_exp = experiment_QC(theta,N,'qasm_simulator','z')
                    writer.writerow([theta,n_rounds,X_exp,Y_exp,Z_exp])
            print('Step: {} / 25'.format(step+1))



acquiring_physical_data = False
if acquiring_physical_data == True:
    x = np.linspace(0,np.pi/2,5)
    with open('data_physical4.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Theta', 'Rounds', '<X>', '<Y>', '<Z>'])

        for step,theta in enumerate(x):
            for n_rounds in range(1,5):
                    N = n_rounds
                    X_exp = experiment_QC(theta,N,'ibmq_ourense','x')
                    Y_exp = experiment_QC(theta,N,'ibmq_ourense','y')
                    Z_exp = experiment_QC(theta,N,'ibmq_ourense','z')
                    writer.writerow([theta,n_rounds,X_exp,Y_exp,Z_exp])
            print('Step: {} / 5'.format(step+1))
