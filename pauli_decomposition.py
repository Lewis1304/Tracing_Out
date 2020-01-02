import numpy as np
from itertools import product

def construct_pauli_strings(m):
    basis = [np.eye(2), np.array([ [0,1],[1,0] ]), np.array([ [0,-1j],[1j,0] ]), np.array([ [1,0],[0,-1] ])]
    basis_labels = ['I','X','Y','Z'] #pauli matrices and identity
    if m == 1:
        return basis,basis_labels

    else:
        initial = []
        initial_labels = []
        ranges = [basis for i in range(m)]
        ranges_labels = [basis_labels for i in range(m)]

        for i,j in zip(product(*ranges),product(*ranges_labels)):
            initial.append(i)

            dummy = ''
            for q in range(len(j)):
                dummy += j[q]
            initial_labels.append(dummy)

        pauli_strings = []
        for i in range(4**m):
            dump = [np.kron(initial[i][0],initial[i][1])]
            for j in range(2,m):
                dump.append(np.kron(dump[-1],initial[i][j]))
            pauli_strings.append(dump[-1])

        return pauli_strings,initial_labels

def construct_matrix(m,H):
    strings,labels = construct_pauli_strings(m)
    H_pauli = sum(np.trace(strings[i] @ H) * strings[i] for i in range(4**m)) / 2**m

    decomposition = []
    for i in range(4**m):
        if np.trace(strings[i] @ H) != 0:
            decomposition.append('{}*{}'.format(np.trace(strings[i] @ H),labels[i]))

    decomposition_string = '{}'.format(decomposition[0])
    for i in decomposition[1:]:
        decomposition_string += ' + {}'.format(i)

    return H_pauli, decomposition_string


m = 2    #number of qubits
H = np.array([ [1,0,0,0],[0,np.exp(0.25j*np.pi),0,0],[0,0,np.exp(0.5j*np.pi),0],[0,0,0,np.exp(0.75j*np.pi)]]) #controlled -iY example
H_pauli,decomposition_string = construct_matrix(m,H)
print(H_pauli)
print(decomposition_string)
