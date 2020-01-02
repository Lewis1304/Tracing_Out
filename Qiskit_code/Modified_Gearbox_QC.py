from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.tools.monitor import job_monitor

def Initial_circuit(theta):
    # Build the initial-sub-circuit

    #Controlled -iY Gate
    U_small = np.kron(np.diag([0,1]),[[0,-1],[1,0]]) + np.diag([1,1,0,0])
    cU = Operator(U_small)

    initial_sub_q = QuantumRegister(2)
    initial_sub_circ = QuantumCircuit(initial_sub_q, name='initial_sub_circ')

    initial_sub_circ.ry(2 * theta,initial_sub_q[0])
    initial_sub_circ.unitary(cU, [initial_sub_q[1],initial_sub_q[0]], label='cU')
    initial_sub_circ.ry(-2 * theta,initial_sub_q[0])
    initial_sub_circ.cry(np.pi / 2,initial_sub_q[0],initial_sub_q[1])

    return initial_sub_circ.to_instruction()

#Controlled-Controlled -iY Gate, ccz * ccx = cc(-iy)
def custom_gate():

    Z = np.array([[1,0],[0,-1]])
    I = np.eye(2)

    qubits = QuantumRegister(3)
    circuit = QuantumCircuit(qubits)

    #ccz gate
    V = (1-1j)*(I + 1j*Z)*0.5
    c_v_array = np.kron(np.diag([0,1]),V) + np.diag([1,1,0,0])
    c_v = Operator(c_v_array)

    c_v_array_dagger = np.kron(np.diag([0,1]),V.conj().T) + np.diag([1,1,0,0])
    c_v_dagger = Operator(c_v_array_dagger)

    circuit.unitary(c_v, [qubits[2],qubits[1]], label='c_v')
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(c_v_dagger, [qubits[2],qubits[1]], label='c_v_dagger')
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(c_v, [qubits[2],qubits[0]], label='c_v')

    #ccx gate
    circuit.ccx(qubits[0],qubits[1],qubits[2])

    return circuit.to_instruction()

def Subsequent_circuit(theta):
    # Build subsequent sub-circuit
    sub_q = QuantumRegister(3)
    sub_circ = QuantumCircuit(sub_q, name='sub_circ')
    sub_circ.ry(2 * theta,sub_q[1])
    sub_circ.append(custom_gate(),[sub_q[0],sub_q[1],sub_q[2]])
    sub_circ.ry(-2 * theta,sub_q[1])
    if N > 2:
        sub_circ.cry(np.pi/2,sub_q[1],sub_q[2]) #reset gate
    return sub_circ.to_instruction()

def experiment_QC(theta,N,name):

    register_qubits = QuantumRegister(N)
    state_qubit = QuantumRegister(1)
    circ = QuantumCircuit(register_qubits,state_qubit)

    RUS_initial = Initial_circuit(theta)
    RUS = Subsequent_circuit(theta)

    circ.append(RUS_initial, [register_qubits[0],state_qubit])

    for i in range(1,N):
        circ.append(RUS, [register_qubits[i-1],register_qubits[i],state_qubit])
    circ.measure_all()

    #Simulate Circuit
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    qcomp = provider.get_backend(name)

    """
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend = simulator,shots = 8192)
    result = job.result()
    success_states = ['0'+'0'*(N-i) + '1'*i for i in range(N)] + ['1'+'0'*(N-i) + '1'*i for i in range(N)]
    print(np.sum([result.get_counts(circ)[i] for i in success_states])/8192)
    print(result.get_counts(circ))
    """

    #stats about the circuit implemented on the actual Quantum Computer- Different to theoretical circuit!:
    t_circ = transpile(circ, backend=qcomp)
    print('Circuit Depth: ',t_circ.depth())
    print('Circuit Width: ',t_circ.width())
    print('Total Number of Operations: ',t_circ.size())
    print('Amount of paralell subcircuits: ',t_circ.num_tensor_factors())

    job = execute(circ, backend = qcomp,shots = 8192)
    print(job_monitor(job))
    result = job.result()
    result_dictionary = result.get_counts(circ)

    success_states = ['0'+'0'*(N-i) + '1'*i for i in range(N)] + ['1'+'0'*(N-i) + '1'*i for i in range(N)]
    Success_fidelity = np.sum([result_dictionary[i] for i in success_states])/8192

    return Success_fidelity

N = 2
names = ['ibmq_5_yorktown','ibmq_essex','ibmq_ourense','ibmq_vigo','ibmq_burlington']
fidelities = []
for i in names:
    round = experiment_QC(np.pi/4,N,i)
    fidelities.append(round)
    print(i)
    print(round)
print(fidelities)
#fidelities_results = [0.6309814453125, 0.5810546875, 0.7349853515625, 0.7135009765625, 0.601806640625]
#should be 0.75
