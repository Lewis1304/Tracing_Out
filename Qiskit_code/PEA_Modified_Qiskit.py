from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity,basis_state


def PEA_circuit_initial():
    initial_sub_q = QuantumRegister(t + N)
    initial_sub_circ = QuantumCircuit(initial_sub_q, name='initial PEA circ')

    initial_sub_circ.h(initial_sub_q)
    if N ==1:
        for i,q in enumerate(initial_sub_q[:-N][::-1]):
            initial_sub_circ.unitary(Single_Controlled_Unitary(phases,i), [initial_sub_q[-N:],q], label='U**(2**{}).format(i)')

    if N ==2:
        for i,q in enumerate(initial_sub_q[:t][::-1]):
            initial_sub_circ.unitary(Two_Controlled_Unitary(phases,i), [*initial_sub_q[-N:],q], label='U**(2**{}).format(i)')

    qft_dagger(initial_sub_circ,initial_sub_q[:-N],t)

    return initial_sub_circ.to_instruction()

def PEA_circuit_conditional():
    cond_PEA_qubits = QuantumRegister(t + N + 1) #register, state, or
    cond_PEA_circ = QuantumCircuit(cond_PEA_qubits,name = 'cond PEA circ')

    cond_PEA_circ.h(cond_PEA_qubits[:t])
    cond_PEA_circ.ch(cond_PEA_qubits[-1],cond_PEA_qubits[t:-1])

    if N ==1:
        for i,q in enumerate(cond_PEA_qubits[:t][::-1]):
            cond_PEA_circ.unitary(Single_Controlled_Controlled_Unitary(phases,i), [cond_PEA_qubits[t:-1],q,cond_PEA_qubits[-1]], label='U**(2**{}).format(i)')

    if N ==2:
        for i,q in enumerate(cond_PEA_qubits[:t][::-1]):
            cond_PEA_circ.unitary(Two_Controlled_Controlled_Unitary(phases,i), [*cond_PEA_qubits[t:-1],q,cond_PEA_qubits[-1]], label='U**(2**{}).format(i)')


    qft_dagger(cond_PEA_circ,cond_PEA_qubits[:t],t)

    return cond_PEA_circ.to_instruction()



def Single_Controlled_Unitary(phases,power):
    U = np.kron( np.diag([0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]) )
    U += np.diag([1,1,0,0])
    return Operator(U**(2**power))

def Single_Controlled_Controlled_Unitary(phases,power):
    U = np.kron(np.diag([0,0,0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]))
    U += np.diag([1,1,1,1,1,1,0,0])
    return Operator(U**(2**power))

def Two_Controlled_Unitary(phases,power):
    U = np.kron( np.diag([0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]) )
    U += np.diag([1,1,1,1,0,0,0,0])
    return Operator(U**(2**power))

def Two_Controlled_Controlled_Unitary(phases,power):
    U = np.kron(np.diag([0,0,0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]))
    U += np.diag([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0])
    return Operator(U**(2**power))

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


def experiment(p,t,phases):

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

    #print(circuit.decompose())

    #Simulate Circuit
    backend_sim = BasicAer.get_backend('statevector_simulator')
    result = execute(circuit, backend_sim).result()
    state_vector = result.get_statevector(circuit)
    probability_of_failure = np.real(np.sum((state_vector * state_vector.conj().T)[2**( (p * t) + (p-1)*(t-1) ):]))
    return np.round(probability_of_failure,4)




t = 2 #Number of qubits in the first register
p = 7 #Number of rounds

#1 qubit case
phases = [0,0.75]
N = int( np.log2(len(phases))) # number of state qubits
xint = [int(2*i*t - (t+i-1)) for i in range(1,p+1)]

Success = []
for i in range(1,p+1):
    Success.append(1-experiment(i,t,phases))
    print('{} / {}'.format(i,p))

xint = [int(2*i*t - (t+i-1)) for i in range(1,p+1)]
plt.plot(xint,Success,label = '1 qubit Hamiltonian')
print(Success)

#2 qubit case
phases = [0,0.25,0.5,0.75]
N = int( np.log2(len(phases))) # number of state qubitsy

Success = []
for i in range(1,p+1):
    Success.append(1-experiment(i,t,phases))
    print('{} / {}'.format(i,p))
print(Success)

plt.plot(xint,Success,label = '2 qubit Hamiltonian')

plt.xticks(xint)
plt.axhline(1,linestyle = 'dashed',color = 'black')
plt.xlim([t, 2*p*t - (t+p-1)])
plt.xlabel('$Number \ of \ Ancillae$')
plt.ylabel('$Groundstate \ Fidelity, \ \ F(|\\psi_G\\rangle,|\\psi_N\\rangle)$')
plt.legend()
plt.savefig('PEA_Modified_Qiskit_Results.pdf')
plt.show()
