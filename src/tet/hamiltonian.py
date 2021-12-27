import numpy as np
import sys

def createHamiltonian(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N):
    '''
    Creates the hamiltonian of the dimer system
    INPUTS:
        chiA: int(), the nonlinearity parameter of the acceptor,
        chiD: int(), the nonlinearity parameter of the donor,
        coupling_lambda: float(), the coupling parameter between the sites,
        omegaA: float(), frequency of the acceptor oscillator,
        omegaD: float(), frequency of the donor oscillator,
        max_N: int(), the maximum number of bosons at the donor site
    OUTPUTS:
        H: np.array(), shape == ((max_N + 1, max_N + 1)), the hamiltonian of the system
    '''
    
    H = np.zeros((max_N + 1, max_N + 1), dtype=float)

    while True:
        query = input('Is the system coupled? [y/n] ')
        fl_1 = query[0].lower() 
        if query == '' or not fl_1 in ['y','n']: 
            print('Please answer with yes or no')
        else: break

    if fl_1 == 'y': 
        # i bosons at the donor
        for i in range(max_N + 1):
            for j in range(max_N + 1):
                # First term from interaction
                if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (max_N - i))
                # Second term from interaction
                if i == j + 1: H[i][j] = -coupling_lambda * np.sqrt(i * (max_N - i + 1))
                # Term coming from the two independent Hamiltonians
                if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

    if fl_1 == 'n':
        for i in range(max_N + 1):
            for j in range(max_N + 1):
                # Term coming from the two independent Hamiltonians
                if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

    return H