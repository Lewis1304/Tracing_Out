import numpy as np
import cirq
import matplotlib.pyplot as plt

class QftInverse(cirq.Gate):
    """Quantum gate for the inverse Quantum Fourier Transformation
    """

    def __init__(self, num_qubits):
        super(QftInverse, self)
        self._num_qubits = num_qubits

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        """A quantum circuit (QFT_inv) with the following structure.

        ---H--@-------@--------@----------------------------------------------
              |       |        |
        ------@^-0.5--+--------+---------H--@-------@-------------------------
                      |        |            |       |
        --------------@^-0.25--+------------@^-0.5--+---------H--@------------
                               |                    |            |
        -----------------------@^-0.125-------------@^-0.25------@^-0.5---H---

        The number of qubits can be arbitrary.
        """

        qubits = list(qubits)
        while len(qubits) > 0:
            q_head = qubits.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qubits):
                yield (cirq.CZ**(-1/2.0**(i+1)))(qubit, q_head)

    def __str__(self):
        return '-QFT_inv'
"""
class unitary_gate(cirq.Gate):

    def __init__(self, num_qubits, total_phase,power):
        super(unitary_gate, self)
        self._num_qubits = num_qubits
        self.total_phase = total_phase
        self.power = power

    def num_qubits(self):
        return self._num_qubits


    def _unitary_(self):
        #print([excited_phase_converter(i)/(2*np.pi) for i in self.excited_phase])
        #raise Exception
        U = np.array(np.diag([ np.exp(-1j * 2 * np.pi * phase_converter(i)) for i in self.total_phase]))
        #U = np.array(np.diag([np.exp(-1j*2*np.pi*k/4) for k in range(4)]))
        return U**(2**self.power)

    def matrix(self):
        gate = cirq.Gate(matrix = self._unitary_)
        return gate

    def __str__(self):
        return '-U^(2^{})'.format(self.power)
"""

def two_qubit(power,total_phase):
        phases = [phase_converter(i) for i in total_phase]
        U = np.array(np.diag([np.exp(2*np.pi*1.0j*i) for i in phases]))**(2**power)
        gate = cirq.TwoQubitMatrixGate(
            matrix = U)
        return gate

def one_qubit(power,total_phase):
        phases = [phase_converter(i) for i in total_phase]
        U = np.array(np.diag([np.exp(2*np.pi*1.0j*i) for i in phases]))**(2**power)
        #U = np.array([[0,1],[1,0]])
        gate = cirq.SingleQubitMatrixGate(
            matrix = U)
        return gate

class PEA(cirq.Gate):

    def __init__(self, num_qubits, total_phase):
        super(PEA, self)
        self._num_qubits = num_qubits
        self.total_phase = total_phase

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self, qubits):
        yield cirq.H.on_each(*qubits[:-N])

        yield cirq.H.on_each(*qubits[-N:]) #Initializing/scrambling Unitary

        if N ==1:
            for i,q in enumerate(qubits[:-N][::-1]):
                yield one_qubit(i,total_phase)(*qubits[-N:]).controlled_by(q)
        elif N==2:
            for i,q in enumerate(qubits[:-N][::-1]):
                yield two_qubit(i,total_phase)(*qubits[-N:]).controlled_by(q)


        yield QftInverse(self._num_qubits-N)(*qubits[:-N])

    def __str__(self):
        return 'PEA'

class Or_gate(cirq.Gate):

    def __init__(self, num_qubits, precision):
        super(Or_gate, self)
        self._num_qubits = num_qubits
        self.precision = precision

    def num_qubits(self):
        return self._num_qubits

    def _decompose_(self,qubits):
        yield cirq.X.on_each(*qubits)

        for i in range(self.precision-1):
            if i != self.precision - 2:
                yield cirq.TOFFOLI.on(*qubits[2*i:2*(i+1)], qubits[self.precision+i])
                yield cirq.X.on(qubits[self.precision+i])
            else:
                yield cirq.TOFFOLI.on(*qubits[2*i:2*(i+1)], qubits[self.precision+i])


    def __str__(self):
        return 'OR'


def phase_converter(phase):
    decimal = 0
    for step,i in enumerate(phase):
        decimal += int(i) * (2**-(step + 1))
    return decimal

def display_energy_levels(total_phase):
    plt.clf()
    energy_levels = [phase_converter(i) for i in total_phase]
    for i in energy_levels:
        plt.axhline(i)
    plt.show()


def experiment(t,p,N,total_phase,print_circuit):

    PEA_gate = PEA(t+N,total_phase) # create GateOperation with correct n_qubits

    n_register_qubits = p*t
    n_state_qubits = N
    n_or_qubits = (p-1) * (t-1)
    register_qubits = [cirq.GridQubit(0, i) for i in range(n_register_qubits)]
    or_qubits = [cirq.GridQubit(1,i) for i in range(n_or_qubits)]
    state_qubits = [cirq.GridQubit(2, i) for i in range(n_state_qubits)] # use grid to separate register and state qubits


    ## Build Circuit

    circuit = cirq.Circuit()
    circuit.append(PEA_gate(*register_qubits[:t], *state_qubits)) # first step of the algorithm. Use first ancillae first


    for i in range(1, p):

        if t == 1:
            circuit.append(PEA_gate(*register_qubits[i*t:(i+1)*t],     # Use next batch of ancillae
                           *state_qubits).controlled_by(register_qubits[i-1]))
        else:
            circuit.append(Or_gate(2*t -1,t)(*register_qubits[(i-1)*t:i*t], *or_qubits[(i - 1) * (t - 1): i * (t-1)]))
            circuit.append(PEA_gate(*register_qubits[i*t:(i+1)*t],
                           *state_qubits).controlled_by(or_qubits[i * (t-1) -1]))

    if print_circuit:
        print('Full Circuit:\n',circuit)
        first_moment = cirq.Circuit.from_ops(cirq.decompose_once(cirq.decompose_once(cirq.Circuit((circuit[0],)))[0]))
        print('First application of PEA:\n', first_moment)
        if p > 1:
            second_moment = cirq.Circuit.from_ops(cirq.decompose_once(cirq.decompose_once(cirq.Circuit((circuit[2],)))[0]))
            print('Second application of PEA:\n', second_moment)

    total_qubits = register_qubits  + or_qubits + state_qubits

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit,qubit_order = total_qubits[::-1])
    state_vector = result.final_state
    prob_of_failure  = np.real(np.sum((state_vector * state_vector.conj().T)[2**( (p * t) + (p-1)*(t-1) ):]))
    return np.round(prob_of_failure,4)



p = 7 # number of rounds
t = 2 # number of ancillae
total_phase = ['0','01']
N = int( np.log2(len(total_phase))) # number of state qubits

Success = []
for i in range(1,p+1):
    Success.append(1-experiment(t,i,N,total_phase,print_circuit= False))
    print('{} / {}'.format(i,p))
xint = [int(2*i*t - (t+i-1)) for i in range(1,p+1)]
plt.plot(xint,Success,label = '1 qubit Hamiltonian')
print(Success)


total_phase = ['0','01','10','11']
N = int( np.log2(len(total_phase))) # number of state qubits
Success = []
for i in range(1,p+1):
    Success.append(1-experiment(t,i,N,total_phase,print_circuit= False))
    print('{} / {}'.format(i,p))
print(Success)

plt.plot(xint,Success,label = '2 qubit Hamiltonian')
plt.xticks(xint)
plt.axhline(1,linestyle = 'dashed',color = 'black')
plt.xlim([t, 2*p*t - (t+p-1)])
plt.xlabel('$Number \ of \ Ancillae$')
plt.ylabel('$Groundstate \ Fidelity, \ \ F(|\\psi_G\\rangle,|\\psi_N\\rangle)$')
plt.legend()
plt.savefig('Proposal_PEA.pdf')
plt.show()
