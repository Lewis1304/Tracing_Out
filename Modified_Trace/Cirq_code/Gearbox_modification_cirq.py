import numpy as np
import cirq
import matplotlib.pyplot as plt

class RUS(cirq.Gate):

    def __init__(self, num_qubits, theta):
        super(RUS, self)
        self._num_qubits = num_qubits
        self.theta = theta

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        y = y_gate(1)
        yield cirq.Ry(2 * self.theta)(*qubits[:-1])
        yield y.controlled_by(*qubits[:-1])(qubits[-1])
        yield cirq.Ry(-2 * self.theta)(*qubits[:-1])
        yield cirq.Ry(np.pi / 2).controlled_by(*qubits[:-1])(qubits[-1])


    def __str__(self):
        return '-RUS'

class y_gate(cirq.Gate):

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self):
            return self._num_qubits

        def _unitary_(self):
            return np.array([
            [0, -1.0],
            [1.0, 0]
            ])

        def _circuit_diagram_info_(self,args):
            return '-iY'


def failure(theta,n):
    p_theta = np.cos(theta)**4 + np.sin(theta)**4
    return (1-p_theta)**n #Somehow we are missing a factor of 2^(n-1)!!!


def experiment(N,theta,print_circuit):
    RUS_gate = RUS(2,theta)

    register_qubits = [cirq.GridQubit(0,i) for i in range(N)]
    state_qubit = cirq.GridQubit(1,0)

    circuit = cirq.Circuit()
    circuit.append(RUS_gate(register_qubits[0],state_qubit))

    for i in range(1,N):
        circuit.append(RUS_gate(register_qubits[i],state_qubit).controlled_by(register_qubits[i-1]))

    if print_circuit:
        print('Full Circuit:\n',circuit)
        first_moment = cirq.Circuit.from_ops(cirq.decompose_once(cirq.decompose_once(cirq.Circuit((circuit[0],)))[0]))
        print('First application of RUS:\n', first_moment)

        if N > 1:
            second_moment = cirq.Circuit.from_ops(cirq.decompose_once(cirq.decompose_once(cirq.Circuit((circuit[1],)))[0]))
            print('Second application of RUS:\n', second_moment)


    total_qubits = register_qubits + [state_qubit]
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit,qubit_order = total_qubits)
    state_vector = result.final_state
    #print(result)
    #print(state_vector)
    H = np.kron(np.diag( np.eye(2**(N))[-1]),np.eye(2))
    prob  = np.real(state_vector.conj().T @ H @ state_vector)
    #print('Simulated probability:', np.round(prob,4))
    #print('Analytic probability:',np.round(failure(theta,N),4))
    return np.round(prob,4)


N = 12 #number of rounds

experiment(2,np.pi/6,print_circuit = True)

theta_list = [np.pi/4,np.pi / 6,np.pi/8]
for j,k in zip(theta_list,[4,6,8]):
    theta = j
    success_prob = []
    analytic = []
    for i in range(1,N+1):
        success_prob.append(1-experiment(i,theta,print_circuit = False))
        analytic.append(1-failure(theta,i))
        #print(1-failure(theta,i))

    plt.plot(range(1,N+1),success_prob,label = '$\\theta = \pi / {}$'.format(k))
    #plt.plot(range(1,N),analytic,linestyle = 'dashed')

plt.ylabel('$ \ Fidelity, \ \ F(R|\\psi\\rangle,|\\psi\\rangle)$')
plt.xlabel('$Number \ of \ Ancillae$')
plt.axhline(1,linestyle = 'dashed',color = 'black')
plt.legend()
plt.xlim([1,N])
plt.savefig('Modified_Gearbox_Cirq.pdf')
plt.show()
