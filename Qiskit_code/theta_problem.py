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

def experiment_QC(theta,N,name,measure_type):

    register_qubits = QuantumRegister(N)
    state_qubit = QuantumRegister(1)
    c = ClassicalRegister(1)
    circ = QuantumCircuit(register_qubits,state_qubit,c)

    RUS_initial = Initial_circuit(theta)
    RUS = Subsequent_circuit(theta)

    circ.append(RUS_initial, [register_qubits[0],state_qubit])

    for i in range(1,N):
        circ.append(RUS, [register_qubits[i-1],register_qubits[i],state_qubit])

    if measure_type == 'x':
        circ.h(register_qubits)
        circ.h(state_qubit)
    elif measure_type == 'y':
        circ.sdg(register_qubits)
        circ.sdg(state_qubit)
        circ.h(register_qubits)
        circ.h(state_qubit)

    circ.measure(state_qubit,c)


    if name == 'qasm_simulator':
        simulator = Aer.get_backend(name)
        job = execute(circ, backend = simulator,shots = 10000)
        result = job.result()
        result_dictionary = result.get_counts(circ)

        probs = {}
        for output in ['0','1']:
            if output in result_dictionary:
                probs[output] = result_dictionary[output]/10000
            else:
                probs[output] = 0
        return probs['0'] -  probs['1']

    else:
        #Simulate Circuit
        IBMQ.load_account()
        provider = IBMQ.get_provider('ibm-q')
        qcomp = provider.get_backend(name)

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

"""
N = 2 #number of rounds
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

for theta in np.linspace(0,np.pi/4,4):
    for n_rounds in [2]:
        N = n_rounds
        print(experiment_QC(theta,N,'ibmq_ourense'))
"""


def theoretical(theta):
    return np.arctan(np.tan(theta)**2)

N = 6
x = np.linspace(0,np.pi/2,50)
z_angle = []
x_angle = []
x_y_angle = []
for step,theta in enumerate(x):
    Z_exp = experiment_QC(theta,N,'qasm_simulator','z')
    X_exp = experiment_QC(theta,N,'qasm_simulator','x')
    Y_exp = experiment_QC(theta,N,'qasm_simulator','y')
    print('Step: {} / 50'.format(step))
    #print('theoretical: ',theoretical(theta))
    #print('z_angle: ',np.arccos(Z_exp)/2) #factor of two since every angle is multiplied by 2 on the bloch sphere
    z_angle.append(np.arccos(Z_exp)/2)
    x_angle.append(np.arcsin(X_exp)/2)
    x_y_angle.append(np.arctan(X_exp / Z_exp)/2)


plt.plot(x,z_angle,label = 'arccos(<Z>)')
plt.plot(x,x_angle,label = 'arcsin(<X>)')
plt.plot(x,x_y_angle,label = 'arctan(<X>/<Z>)')
plt.plot(x,[theoretical(theta) for theta in x],label = 'theoretical')
plt.legend()
plt.savefig('theta.pdf')
plt.show()

#data = open('data.txt','w')
#data.write('Angle:\t # Rounds\t Angle ')