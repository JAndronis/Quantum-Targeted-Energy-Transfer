import numpy as np
import sys

def createHamiltonian(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, coupled=True):
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
    
    H = np.zeros((max_N + 1, max_N + 1), dtype=float)

    if coupled: 
        # i bosons at the donor
        for i in range(max_N + 1):
            for j in range(max_N + 1):
                # First term from interaction
                if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (max_N - i))
                # Second term from interaction
                if i == j + 1: H[i][j] = -coupling_lambda * np.sqrt(i * (max_N - i + 1))
                # Term coming from the two independent Hamiltonians
                if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

    else:
        for i in range(max_N + 1):
            for j in range(max_N + 1):
                # Term coming from the two independent Hamiltonians
                if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

    return H

def createHamiltonians(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N):
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

    H, H_uncoupled = np.zeros((max_N + 1, max_N + 1), dtype=float), np.zeros((max_N + 1, max_N + 1), dtype=float)

    # i bosons at the donor
    for i in range(max_N + 1):
        for j in range(max_N + 1):
            # First term from interaction
            if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (max_N - i))
            # Second term from interaction
            if i == j + 1: H[i][j] = -coupling_lambda * np.sqrt(i * (max_N - i + 1))
            # Term coming from the two independent Hamiltonians
            if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

    for i in range(max_N + 1):
        for j in range(max_N + 1):
            # Term coming from the two independent Hamiltonians
            if i == j: H_uncoupled[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2
    
    return H, H_uncoupled