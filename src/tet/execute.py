import numpy as np
import shutil
from tet import Hamiltonian, writeData, AvgBosons

def execute(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, max_t, data_dir, zip_files=False):
    problemHamiltonian = Hamiltonian.Hamiltonian(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N).createHamiltonian()
    eigenvalues, eigenvectors = np.linalg.eigh(problemHamiltonian)

    initial_state = np.zeros(max_N+1,dtype=float)
    for n in range(max_N+1): initial_state[n] = np.exp(-(max_N-n)**2)
    initial_state = initial_state / np.linalg.norm(initial_state)

    avg_ND_analytical = AvgBosons.AvgBosons(max_t=max_t,
                                            max_N=max_N, 
                                            eigvecs=eigenvectors,
                                            eigvals=eigenvalues,
                                            initial_state=initial_state).computeAverage()                                              
    title_file = f'ND_analytical-λ={coupling_lambda}-t_max={max_t}-χA={chiA}-χD={chiD}.txt'
    writeData.writeData(data=avg_ND_analytical, destination=data_dir, name_of_file=title_file)