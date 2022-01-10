import numpy as np

class AvgBosons:
    def __init__(self, max_t, max_N, eigvecs, eigvals, initial_state):
        self.max_t = max_t
        self.max_N = max_N
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        self.initial_state = initial_state
        
        if np.ndim(eigvals)>1:
            self.coeff_c = np.zeros((self.eigvecs.shape[0],self.max_N+1), dtype=float)
            self.coeff_b = self.eigvecs
            for i in range(len(self.eigvecs)):
                for j in range(self.max_N+1):
                    self.coeff_c[i][j] = np.vdot(self.eigvecs[i][:,j], self.initial_state)
        else:
            self.coeff_c = np.zeros(self.max_N+1, dtype=float)
            for i in range(self.max_N+1): 
                self.coeff_c[i] = np.vdot(self.eigvecs[:,i], self.initial_state)

            self.coeff_b = self.eigvecs
    
    def computeAverage(self):
        '''
        Function that calculates the average number of bosons in the dimer system.
        The calculation is done based on the equation (25) in our report.

        INPUTS:
        max_t == int, 
                    The maximum time of the experiment.
        max_N == int, 
                    The number of bosons on the donor site.
        eigvecs == np.array,
                    shape == (max_N+1, max_N+1),
                    The eigenvectors of the hamiltonian of the system as columns in a numpy array.
        eigvals == np.array,
                    shape == max_N+1,
                    The eigenvalues of the hamiltonian of the system.
        initial_state == np.array,
                            shape == max_N+1,
                            An initial state for the system defined as the normalised version of:
                            np.exp(-(max_N-n)**2) where n is the n-th boson at the donor site.

        OUTPUTS:
        avg_N == list,
                    len == max_t
                    The average number of bosons at the donor.
        '''

        avg_N = []
        _time = range(0, self.max_t+1)

        for t in _time:
            avg_N.append(np.real(self._computeAverageCalculation(t)))

        return avg_N

    def _computeAverageCalculation(self, t):
        sum_j = 0
        for j in range(self.max_N+1):
            sum_i = sum(self.coeff_c*self.coeff_b[j,:]*np.exp(-1j*self.eigvals*t))
            sum_k = sum(self.coeff_c.conj()*self.coeff_b[j,:].conj()*np.exp(1j*self.eigvals*t)*sum_i)
            sum_j += sum_k*j
        return sum_j
    
    def computeAverageComb(self):
        '''
        Function that calculates the average number of bosons in the dimer system.
        The calculation is done based on the equation (25) in our report.

        INPUTS:
        max_t == int, The maximum time of the experiment.
        max_N == int, The number of bosons on the donor site.
        eigvecs == np.array,
                    shape == (max_N+1, max_N+1),
                    The eigenvectors of the hamiltonian of the system as columns in a numpy array.
        eigvals == np.array,
                    shape == max_N+1,
                    The eigenvalues of the hamiltonian of the system.
        initial_state == np.array,
                            shape == max_N+1,
                            An initial state for the system defined as the normalised version of:
                            np.exp(-(max_N-n)**2) where n is the n-th boson at the donor site.

        OUTPUTS:
        avg_N == list,
                    len == max_t
                    The average number of bosons at the donor.
        '''

        _time = range(self.max_t+1)
        avg_N = np.zeros((self.eigvals.shape[0], len(_time)))
        
        for t in _time:
            avg_N[:,t] = np.real(self._computeAverageCalculationComb(t))

        return avg_N

    def _computeAverageCalculationComb(self, t):
        sum_j = 0
        for j in range(self.max_N+1):
            sum_i = np.sum(self.coeff_c*self.coeff_b[:,j,:]*np.exp(-1j*self.eigvals*t), axis=1)
            sum_k = np.sum((self.coeff_c.conj()*self.coeff_b.conj()[:,j,:]*np.exp(1j*self.eigvals*t))*sum_i.reshape(-1,1), axis=1)
            sum_j += j*sum_k
        return sum_j