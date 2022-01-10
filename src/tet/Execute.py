import numpy as np
from itertools import product
from tet.Hamiltonian import Hamiltonian
from tet.AvgBosons import AvgBosons
from tet.data_process import writeData

class Execute:
    def __init__(self, chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, max_t, data_dir, return_data=False):
        self.chiA = chiA
        self.chiD = chiD
        self.coupling_lambda = coupling_lambda
        self.omegaA = omegaA
        self.omegaD = omegaD
        self.max_N = max_N
        self.max_t = max_t
        self.data_dir = data_dir
        self.return_data = return_data

        self.initial_state = np.zeros(max_N+1,dtype=float)
        for n in range(max_N+1): self.initial_state[n] = np.exp(-(max_N-n)**2)
        self.initial_state = self.initial_state / np.linalg.norm(self.initial_state)

    def __call__(self):
        if np.ndim(self.chiA)>0: return self.executeGrid()
        else: return self.executeOnce()

    def executeGrid(self):
        H = np.zeros((len(list(product(self.chiA, self.chiD))), self.max_N+1, self.max_N+1))
        eigenvalues = np.zeros((len(list(product(self.chiA, self.chiD))), self.max_N+1))
        eigenvectors = np.zeros((len(list(product(self.chiA, self.chiD))), self.max_N+1, self.max_N+1))

        avg_ND_analytical = np.zeros((eigenvalues.shape[0], 2001))

        counter = 0
        param_id = []
        for combination in product(self.chiA, self.chiD):
            x = combination[0]
            y = combination[1]
            H[counter] = Hamiltonian(x, y, self.coupling_lambda, self.omegaA, self.omegaD, self.max_N).createHamiltonian()
            eigenvalues[counter], eigenvectors[counter] = np.linalg.eigh(H[counter])
            counter += 1
            param_id.append(combination)
            
        avg_ND_analytical = AvgBosons(max_t=self.max_t,
                                      max_N=self.max_N,
                                      eigvecs=eigenvectors,
                                      eigvals=eigenvalues,
                                      initial_state=self.initial_state).computeAverageComb()
        
        if self.return_data: return avg_ND_analytical
        else:
            counter_2 = 0
            for combination in product(self.chiA, self.chiD):
                i = combination[0]
                j = combination[1]
                title = f'ND_analytical-λ={self.coupling_lambda}-t_max={self.max_t}-χA={i}-χD={j}.txt'
                _tmp_data = avg_ND_analytical[counter_2]
                writeData(_tmp_data, self.data_dir, title)
                counter_2 += 1
    
    def executeOnce(self):
        problemHamiltonian = Hamiltonian(self.chiA, self.chiD, self.coupling_lambda, self.omegaA, self.omegaD, self.max_N).createHamiltonian()
        eigenvalues, eigenvectors = np.linalg.eigh(problemHamiltonian)

        avg_ND_analytical = AvgBosons(max_t=self.max_t,
                                      max_N=self.max_N, 
                                      eigvecs=eigenvectors,
                                      eigvals=eigenvalues,
                                      initial_state=self.initial_state).computeAverage()                                              
        
        if self.return_data: return avg_ND_analytical
        else:
            title_file = f'ND_analytical-λ={self.coupling_lambda}-t_max={self.max_t}-χA={self.chiA}-χD={self.chiD}.txt'
            writeData(data=avg_ND_analytical, destination=self.data_dir, name_of_file=title_file)
