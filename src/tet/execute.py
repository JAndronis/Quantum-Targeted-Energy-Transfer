import numpy as np
from tet import Hamiltonian, writeData, AvgBosons

def execute(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, data_dir):
    problemHamiltonian = Hamiltonian.Hamiltonian(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N).createHamiltonian()
    eigenvalues, eigenvectors = np.linalg.eigh(problemHamiltonian)

    initial_state = np.zeros(max_N+1,dtype=float)
    for n in range(max_N+1): initial_state[n] = np.exp(-(max_N-n)**2)
    initial_state = initial_state / np.linalg.norm(initial_state)

    t_max = 2000

    avg_ND_analytical = AvgBosons.AvgBosons(max_t=t_max,
                                            max_N=max_N, 
                                            eigvecs=eigenvectors,
                                            eigvals=eigenvalues,
                                            initial_state=initial_state).computeAverage()                                              
    title_file = f'ND_analytical-λ={coupling_lambda}-χA={chiA}-χD={chiD}.txt'
    writeData.writeData(avg_ND_analytical, destination=data_dir, name_of_file=title_file)