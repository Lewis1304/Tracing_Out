from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity,basis_state
import itertools
import csv
from numpy import linalg as LA
from numpy import random
from scipy.linalg import expm
import matplotlib.ticker as mticker

#Creating U(2)

def granschmit(X): #Creates O out of linearly independent vectors (A)
    Q, R = np.linalg.qr(X)
    return Q

def linear_independence(A): #checks whether A has dependent vectors
    w,v = LA.eig(A)
    for i in w:
        if i == 0: print('DEPENDENT')

def u2(power): #creates U(2) via similarity transformation ODO^-1

    """
    #Diagonal Unitary
    U = np.diag([1,1j,-1,-1j])

    #Constructing unitary for linearly independent vectors
    A = np.array([
        [1, 1, 0, 5],
        [0, 1, 0, 3],
        [1, 0, 1, 1],
        [0, 4, 0, 1]], dtype=float)

    linear_independence(A)
    O = granschmit(A)
    U = O @ np.diag([1,1j,-1,-1j]) @ O.T.conj()
    """
    #Perturbs diagonal Hamiltonian
    lambdas = np.array([0,np.pi/2,np.pi,3*np.pi/2])
    H = np.diag(lambdas) + 2**(-power) * (np.random.rand(4,4))
    H = (H + H.conj().T) / 2
    U = expm(-1j*H)
    return U


def u2_gs(U): #groundstate of hamiltonian. i.e. eigenstate corresponding to eigenvalue 1
    w, v = LA.eig(U)
    return v[:,np.where(np.round(w) == 1)[0][0]]

def Single_Controlled_Unitary(U):
    Q = np.kron( np.diag([0,1]),U )
    Q += np.diag([1,1,1,1,0,0,0,0])
    return Operator(Q)

def Two_Controlled_Unitary(U):
    Q = np.kron( np.diag([0,0,0,1]),U)
    Q += np.diag([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0])
    return Operator(Q)

def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(int(n/2)):
        circ.swap(qubit, n-qubit-1)
    for j in range(n,0,-1):
        k = n - j
        for m in range(k):
            circ.cu1(-math.pi/float(2**(k-m)), n-m-1, n-k-1)
        circ.h(n-k-1)


def experiment_QC(N,p,t,sigma_1,sigma_2,power):

    n_register_qubits =  p*t
    n_state_qubits = N
    n_or_qubits = p * (t-1)

    register_qubits = QuantumRegister(n_register_qubits)
    state_qubits = QuantumRegister(n_state_qubits)
    or_qubits = QuantumRegister(n_or_qubits)
    c = ClassicalRegister(N)

    circ = QuantumCircuit(register_qubits,or_qubits,state_qubits,c)

    circ.h(state_qubits) #initial input state
    """
    for dummy in range(5):
        circ.u3(*(2*np.pi*np.random.rand(3)),state_qubits[0])
        circ.u3(*(2*np.pi*np.random.rand(3)),state_qubits[1])
    """

    circ.h(register_qubits)

    #Initial Round

    #Controlled U(2)
    for i,q in enumerate(register_qubits[:t][::-1]):
        for k in range(0,2**i):
            circ.unitary(Single_Controlled_Unitary(U),[q,*state_qubits][::-1])

    #QFT-dagger
    for qubit in range(int(t/2)):
        circ.swap(qubit, t-qubit-1)
    for j in range(t,0,-1):
        k = t - j
        for m in range(k):
            circ.cu1(-np.pi/float(2**(k-m)), t-m-1, t-k-1)
        circ.h(t-k-1)

    #Reset

    #OR-Gate
    circ.x(register_qubits[:t])
    circ.x(or_qubits[0])
    circ.ccx(register_qubits[0],register_qubits[1],or_qubits[0])

    #Scrambling gate
    circ.ch(or_qubits[0],state_qubits)

    """
    for dummy in range(5):
        circ.cu3(*(2*np.pi*np.random.rand(3)),or_qubits[0],state_qubits[0])
        circ.cu3(*(2*np.pi*np.random.rand(3)),or_qubits[0],state_qubits[1])
    """
    #Subsequent Rounds
    for j in range(1,p):

        #CCU
        for i,q in enumerate(register_qubits[t*j:t*(j+1)][::-1]):
            for k in range(0,2**i):
                circ.unitary(Two_Controlled_Unitary(U),[q,or_qubits[j-1],*state_qubits][::-1])

        #QFT-dagger
        for qubit in range(int(t/2)):
            circ.swap(qubit + t*j, t-qubit-1 + t*j)
        for f in range(t,0,-1):
            k = t - f
            for m in range(k):
                circ.cu1(-np.pi/float(2**(k-m)), t-m-1 + t*j, t-k-1 + t*j)
            circ.h(t-k-1 + t*j)

        #OR-Gate
        circ.x(register_qubits[t*j:t*(j+1)])
        circ.x(or_qubits[j])
        circ.ccx(register_qubits[t*j],register_qubits[t*j + 1],or_qubits[j])

        #Scrambling gate
        circ.ch(or_qubits[j],state_qubits)
        """
        for dummy in range(5):
            circ.cu3(*(2*np.pi*np.random.rand(3)),or_qubits[j],state_qubits[0])
            circ.cu3(*(2*np.pi*np.random.rand(3)),or_qubits[j],state_qubits[1])
        """

    #Pauli Measurements (Very messy I know)
    if sigma_1 == 'z' and sigma_2 == 'I':
        circ.iden(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'x' and sigma_2 == 'I':
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'y' and sigma_2 == 'I':
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'z':
        circ.swap(state_qubits[0],state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'x':
        circ.swap(state_qubits[0],state_qubits[1])
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'y':
        circ.swap(state_qubits[0],state_qubits[1])
        circ.iden(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'z':
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.iden(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.iden(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'z':
        circ.iden(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'z':
        circ.iden(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    circ.measure(state_qubits,c)

    shots = 8192
    backend = Aer.get_backend('qasm_simulator')

    job = execute(circ, backend = backend, shots = shots)
    result = job.result()
    result_dictionary = result.get_counts(circ)
    probs = {}
    for output in ['00','01','10','11']:
        if output in result_dictionary:
            probs[output] = result_dictionary[output]
        else:
            probs[output] = 0

    return (probs['00'] + probs['11'] - probs['01'] - probs['10']) / shots


def groundstate_representation(U):
    I = np.eye(2)
    x = np.array([[0,1],[1,0]])
    y = np.array([[0,-1j],[1j,0]])
    z = np.array([[1,0],[0,-1]])
    gs = u2_gs(U)
    results = []
    a = []
    b = []
    T = []
    for j,k in itertools.product([I,z,x,y],[I,z,x,y]):

        experiment_result = gs.T.conj() @ np.kron(j,k) @ gs
        if np.array_equal(j,I) and np.array_equal(k,I):
            continue

        if np.array_equal(j,I) and (not np.array_equal(k,I)):
            a.append(experiment_result)

        if np.array_equal(k,I) and (not np.array_equal(j,I)):
            b.append(experiment_result)

        if (not np.array_equal(j,I)) and (not np.array_equal(k,I)):
            T.append(experiment_result)

    return a,b,T

def analytic_fidelity(P,n):
    N_star = 1
    N = 2**n
    return 1/N_star * (1 - ((N - N_star) / N)**(P+1) )

def computed_fidelity(a1,b1,T1,U):
    a2, b2, T2 = groundstate_representation(U)
    return 0.25 * (1 + np.dot(a1,a2) + np.dot(b1,b2) + np.dot(T1,T2))

n = 2
t = 2 #Do t = 2 for now
max_rounds = 5
x = range(1,max_rounds+1)
power = [1,2,3,4]
groundstates = []

for pow in power:
    results = []
    U = u2(pow)
    w,v = LA.eig(U)
    average = np.mean(abs(w-np.round(w))) #working out the difference between perturbed U e.v. and non-perturbed.

    for p in x:
        a = []
        b = []
        T = []
        for j,k in itertools.product(['I','z','x','y'],['I','z','x','y']):
            if j == 'I' and k == 'I':
                continue

            experiment_result = experiment_QC(n,p,t,j,k,U)

            if j =='I' and k != 'I':
                a.append(experiment_result)
            if k == 'I' and j != 'I':
                b.append(experiment_result)
            if j != 'I' and k != 'I':
                T.append(experiment_result)

        results.append(computed_fidelity(a,b,T,U))

        print('{} / {}'.format(p,max_rounds))
        #print( np.real(np.round(computed_fidelity(a,b,T,U),4)) )

    plt.plot(x,results,label = '$\\epsilon = 2^{},\\langle|\\lambda_B - \\lambda_A|\\rangle = {} $'.format(-pow,np.round(average,5)))

plt.plot(x,[analytic_fidelity(i,2) for i in x],label = 'analytic',color = 'k')
plt.legend()
#plt.savefig('General_Hamiltonian_Fidelity.pdf')
plt.xlabel('Number of Rounds')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.ylabel('Fidelity')
plt.show()
