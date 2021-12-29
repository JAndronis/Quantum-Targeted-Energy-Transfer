import numpy as np
from itertools import product
from tet import Hamiltonian, data_process, AvgBosons

def execute(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, max_t, data_dir):

    initial_state = np.zeros(max_N+1,dtype=float)
    for n in range(max_N+1): initial_state[n] = np.exp(-(max_N-n)**2)
    initial_state = initial_state / np.linalg.norm(initial_state)

    if np.ndim(chiA)>0:
        H = np.zeros((len(list(product(chiA, chiD))), max_N+1, max_N+1))
        eigenvalues = np.zeros((len(list(product(chiA, chiD))), max_N+1))
        eigenvectors = np.zeros((len(list(product(chiA, chiD))), max_N+1, max_N+1))

        avg_ND_analytical = np.zeros((eigenvalues.shape[0], 2001))

        counter = 0
        param_id = []
        for combination in product(chiA, chiD):
            x = combination[0]
            y = combination[1]
            H[counter] = Hamiltonian.Hamiltonian(x, y, coupling_lambda, omegaA, omegaD, max_N).createHamiltonian()
            eigenvalues[counter], eigenvectors[counter] = np.linalg.eigh(H[counter])
            counter += 1
            param_id.append(combination)
            
        avg_ND_analytical = AvgBosons.AvgBosons(max_t=max_t,
                                                max_N=max_N,
                                                eigvecs=eigenvectors,
                                                eigvals=eigenvalues,
                                                initial_state=initial_state).computeAverageComb()
        
        return avg_ND_analytical

    else:
        problemHamiltonian = Hamiltonian.Hamiltonian(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N).createHamiltonian()
        eigenvalues, eigenvectors = np.linalg.eigh(problemHamiltonian)

        avg_ND_analytical = AvgBosons.AvgBosons(max_t=max_t,
                                                max_N=max_N, 
                                                eigvecs=eigenvectors,
                                                eigvals=eigenvalues,
                                                initial_state=initial_state).computeAverage()                                              
        title_file = f'ND_analytical-λ={coupling_lambda}-t_max={max_t}-χA={chiA}-χD={chiD}.txt'
        data_process.writeData(data=avg_ND_analytical, destination=data_dir, name_of_file=title_file)
        return avg_ND_analytical