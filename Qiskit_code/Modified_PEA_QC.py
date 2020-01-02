from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity,basis_state
from qiskit.tools.visualization import plot_histogram

def PEA_circuit_initial():
    initial_sub_q = QuantumRegister(t + N)
    initial_sub_circ = QuantumCircuit(initial_sub_q, name='initial PEA circ')

    initial_sub_circ.h(initial_sub_q)
    if N ==1:
        for i,q in enumerate(initial_sub_q[:-N][::-1]):
            initial_sub_circ.unitary(Single_Controlled_Unitary(i), [initial_sub_q[-N:],q], label='U**(2**{}).format(i)') #target then control

    if N ==2:
        for i,q in enumerate(initial_sub_q[:t][::-1]):
            initial_sub_circ.append(Two_Controlled_Unitary(i),[q,*initial_sub_q[-N:]]) #Control then Target

    qft_dagger(initial_sub_circ,initial_sub_q[:-N],t)

    return initial_sub_circ.to_instruction()

def PEA_circuit_conditional():
    cond_PEA_qubits = QuantumRegister(t + N + 1) #register, state, or. Assuming 1 or qubit maybe generalised in other funcs.
    cond_PEA_circ = QuantumCircuit(cond_PEA_qubits,name = 'cond PEA circ')

    cond_PEA_circ.h(cond_PEA_qubits[:t])
    cond_PEA_circ.ch(cond_PEA_qubits[-1],cond_PEA_qubits[t:-1])

    if N ==1:
        for i,q in enumerate(cond_PEA_qubits[:t][::-1]):
            cond_PEA_circ.append(Single_Controlled_Controlled_Unitary_Circuit(i), [cond_PEA_qubits[-1],q,cond_PEA_qubits[t:-1]]) #or, register, state
    if N ==2:
        for i,q in enumerate(cond_PEA_qubits[:t][::-1]):
            cond_PEA_circ.append(Two_Controlled_Controlled_Unitary(i),[cond_PEA_qubits[-1],q,*cond_PEA_qubits[t:-1]])

    qft_dagger(cond_PEA_circ,cond_PEA_qubits[:t],t)

    return cond_PEA_circ.to_instruction()

###########################################################################################
#Control U(1) gate
def Single_Controlled_Unitary(power):
    phases = [0,0.75]
    U = np.kron( np.diag([0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]) )
    U += np.diag([1,1,0,0])
    return Operator(U**(2**power))

#Control-Control U(1) gate
 #Fixed for phase = .11 = 0.75
def Single_Controlled_Controlled_Unitary_Circuit(i):
    qubits = QuantumRegister(3)
    circuit = QuantumCircuit(qubits)

    V =  np.array([[1,0],[0,np.exp(0.75j * np.pi)]])**(2**i)
    c_v_array = np.kron(np.diag([0,1]),V) + np.diag([1,1,0,0])
    c_v = Operator(c_v_array)

    c_v_array_dagger = np.kron(np.diag([0,1]),V.conj().T) + np.diag([1,1,0,0])
    c_v_dagger = Operator(c_v_array_dagger)

    circuit.unitary(c_v, [qubits[2],qubits[1]], label='c_v')
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(c_v_dagger, [qubits[2],qubits[1]], label='c_v_dagger')
    circuit.cx(qubits[0],qubits[1])
    circuit.unitary(c_v, [qubits[2],qubits[0]], label='c_v')
    return circuit.to_instruction()
##############################################################################################

##############################################################################################
#Control U(2) and Control-control U(2)
#for fixed phase phases = [0,0.25,0.5,0.75]

def Two_Controlled_Unitary(i):
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)

    if i < 3:
        for k in range(0,2**i):
            qc.cu3(0,np.pi/2,0,q[0],q[1])
            qc.cu3(0,-np.pi/2,-np.pi/2,q[0],q[2])
    else:
        qc.iden(q[1])
        qc.iden(q[2])

    return qc.to_instruction()

def controlled_V(i):
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)

    for i in range(0,2**i):
        qc.cu3(0,3*np.pi/8,-np.pi/8,q[0],q[1])
        qc.cu3(0,-3*np.pi/4,-3*np.pi/4,q[0],q[2])
    #print(qc)
    return qc.to_instruction()

def controlled_V_dagger(i):
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)

    if i < 3:
        for i in range(0,2**i):
            qc.cu3(0,-3*np.pi/8,np.pi/8,q[0],q[1])
            qc.cu3(0,3*np.pi/4,3*np.pi/4,q[0],q[2])
    else:
        qc.iden(q[1])
        qc.iden(q[2])
    #print(qc)
    return qc.to_instruction()

def Two_Controlled_Controlled_Unitary(i):
    q = QuantumRegister(4)
    qc = QuantumCircuit(q)

    qc.append(controlled_V(i),[q[1],q[2],q[3]])
    qc.cx(q[0],q[1])
    qc.append(controlled_V_dagger(i),[q[1],q[2],q[3]])
    qc.cx(q[0],q[1])
    qc.append(controlled_V(i),[q[0],q[2],q[3]])

    return qc.to_instruction()
###################################################################################################

def qft_dagger(circ,q,n):
    qubits = list(q)
    while len(qubits)>0:
        q_head = qubits.pop(0)
        circ.h(q_head)
        for i,qubit in enumerate(qubits):
            circ.cu1(-np.pi/float(2**(i+1)), qubit, q_head)

def or_gate(i):
    qubits = QuantumRegister(2 * t -1)
    or_circ = QuantumCircuit(qubits, name = 'or_circ')

    or_circ.x(qubits)

    for i in range(t-1):
        if i != t-2:
            or_circ.ccx(qubits[2*i],qubits[2*(i+1) -1],qubits[t+i])
            or_circ.x(qubits[t+i])
        else:
            or_circ.ccx(qubits[2*i],qubits[2*(i+1)-1],qubits[t+i])

    return or_circ.to_instruction()

def experiment_QC(p,t):

    n_register_qubits =  p*t
    n_state_qubits = N
    n_or_qubits = (p-1) * (t-1)

    if p == 1:
        register_qubits = QuantumRegister(n_register_qubits)
        state_qubits = QuantumRegister(n_state_qubits)

        circuit = QuantumCircuit(register_qubits,state_qubits)
        circuit.append(PEA_circuit_initial(),[*register_qubits[:t],*state_qubits])

    else:
        register_qubits = QuantumRegister(n_register_qubits,'register')
        or_qubits = QuantumRegister(n_or_qubits,'or')
        state_qubits = QuantumRegister(n_state_qubits,'state')

        circuit = QuantumCircuit(register_qubits,or_qubits,state_qubits)

        circuit.append(PEA_circuit_initial(),[*register_qubits[:t],*state_qubits])
        for i in range(1,p):
            circuit.append(or_gate(i),[*register_qubits[(i-1)*t:i*t], *or_qubits[(i - 1) * (t - 1): i * (t-1)]])
            circuit.append(PEA_circuit_conditional(),[*register_qubits[i*t:(i+1)*t],*state_qubits,or_qubits[i * (t-1) -1]])
    circuit.measure_all()
    """
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    qcomp = provider.get_backend('ibmq_16_melbourne')

    job = execute(circuit, backend = qcomp,shots = 8192)
    from qiskit.tools.monitor import job_monitor
    print(job_monitor(job))
    result = job.result()

    """
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend = simulator,shots = 8192)
    result = job.result()


    result_dictionary = result.get_counts(circuit)
    list_of_states = list(result_dictionary.keys())
    if N == 1:
        Success_fidelity = np.sum( [result_dictionary[i] for i in list_of_states if i[0] == '0']) / 8192 #We only care about the overlap between the output state and groundstate (|0>)
        return Success_fidelity
    if N == 2:
        Success_fidelity = np.sum( [result_dictionary[i] for i in list_of_states if i[0] == '0' and i[1] == '0']) / 8192 #We only care about the overlap between the output state and groundstate (|00>)
        return Success_fidelity


#phases are [0,0.75] and [0,0.25,0.5,0.75] for the 1 and 2 qubit cases respectively.
N = 2 #1 or 2 qubit Hamiltonian
t=2 #number of precision qubits
p=2 #number of rounds
fidelity = experiment_QC(p,t)
print('Success Fidelity: ',fidelity)
#Success Fidelity:  0.595458984375 (N=1, p=2) should be 0.75
#Success Fidelity:  0.3209228515625 (N=2, p=2) should be ~0.46
