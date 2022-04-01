import numpy as np

class Hamiltonian:
    def __init__(self, chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, coupled=True):
        self.chiA = chiA
        self.chiD = chiD
        self.coupling_lambda = coupling_lambda
        self.omegaA = omegaA
        self.omegaD = omegaD
        self.max_N = max_N
        self.coupled = coupled

    def createHamiltonian(self):
        '''
        Creates the hamiltonian of the dimer system
        INPUTS:
            chiA: int(), the nonlinearity parameter of the acceptor,
            chiD: int(), the nonlinearity parameter of the donor,
            coupling_lambda: float(), the coupling parameter between the sites,
            omegaA: float(), frequency of the acceptor oscillator,
            omegaD: float(), frequency of the donor oscillator,
            max_N: int(), the maximum number of bosons at the donor site,
            coupled: boolean(), variable determining if the function will return the coupled
                                or uncoupled hamiltonian
        OUTPUTS:
            H: np.array(), shape == ((max_N + 1, max_N + 1)), the hamiltonian of the system
        '''
        
        H = np.zeros((self.max_N + 1, self.max_N + 1), dtype=float)

        if self.coupled: 
            # i bosons at the donor
            for i in range(self.max_N + 1):
                for j in range(self.max_N + 1):
                    # First term from interaction
                    if i == j - 1: H[i][j] = -self.coupling_lambda * np.sqrt((i + 1) * (self.max_N - i))
                    # Second term from interaction
                    if i == j + 1: H[i][j] = -self.coupling_lambda * np.sqrt(i * (self.max_N - i + 1))
                    # Term coming from the two independent Hamiltonians
                    if i == j: H[i][j] = self.omegaD * i + 0.5 * self.chiD * i ** 2 + self.omegaA * (self.max_N - i) + 0.5 * self.chiA * (self.max_N - i) ** 2

        else:
            for i in range(self.max_N + 1):
                for j in range(self.max_N + 1):
                    # Term coming from the two independent Hamiltonians
                    if i == j: H[i][j] = self.omegaD * i + 0.5 * self.chiD * i ** 2 + self.omegaA * (self.max_N - i) + 0.5 * self.chiA * (self.max_N - i) ** 2

        return H

    def createHamiltonians(self):
        '''
        Creates both the hamiltonians of the coupled and uncoupled system
        INPUTS:
            chiA: int(), the nonlinearity parameter of the acceptor,
            chiD: int(), the nonlinearity parameter of the donor,
            coupling_lambda: float(), the coupling parameter between the sites,
            omegaA: float(), frequency of the acceptor oscillator,
            omegaD: float(), frequency of the donor oscillator,
            max_N: int(), the maximum number of bosons at the donor site,
            coupled: boolean(), variable determining if the function will return the coupled
                                or uncoupled hamiltonian
        OUTPUTS:
            tupple():
                H: np.array(), shape == ((max_N + 1, max_N + 1)), the hamiltonian of the coupled system
                H_uncoupled: np.array(), shape == ((max_N + 1, max_N + 1)), the hamiltonian of the uncoupled system
        '''

        return self.createHamiltonian(self, coupled=True), self.createHamiltonian(self, coupled=False)
