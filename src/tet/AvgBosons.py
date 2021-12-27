import numpy as np

class AvgBosons:
    def __init__(self, max_t, max_N, eigvecs, eigvals, initial_state):
        self.max_t = max_t
        self.max_N = max_N
        self.eigvecs = eigvecs
        self.eigvals = eigvals
        self.initial_state = initial_state
    
    def computeAverage(self):
        '''
        Function that calculates the average number of bosons in the dimer system.
        The calculation is done based on the equation (25) in our report.

        INPUTS:
        max_t == int(), The maximum time of the experiment.
        max_N == int(), The number of bosons on the donor site.
        eigvecs == np.array(),
                    shape == (max_N+1, max_N+1),
                    The eigenvectors of the hamiltonian of the system as columns in a numpy array.
        eigvals == np.array(),
                    shape == max_N+1,
                    The eigenvalues of the hamiltonian of the system.
        initial_state == np.array(),
                            shape == max_N+1,
                            An initial state for the system defined as the normalised version of:
                            np.exp(-(max_N-n)**2) where n is the n-th boson at the donor site.

        OUTPUTS:
        avg_N == list(),
                    len() == max_t
                    The average number of bosons at the donor.
        '''

        coeff_c = np.zeros(self.max_N+1, dtype=float)
        for i in range(self.max_N+1): 
            coeff_c[i] = np.vdot(self.eigvecs[:,i], self.initial_state)

        coeff_b = self.eigvecs

        avg_N = []
        _time = range(0, self.max_t+1)
        avg_min = self.max_N

        for t in _time:
            avg_N.append(np.real(self._computeAverageCalculation(coeff_c, coeff_b, t)))

        return avg_N

    def _computeAverageCalculation(self, coeff_c, coeff_b, t):
        sum_j = 0
        for j in range(self.max_N+1):
            sum_i = sum(coeff_c*coeff_b[j,:]*np.exp(-1j*self.eigvals*t))
            sum_k = sum(coeff_c.conj()*coeff_b[j,:].conj()*np.exp(1j*self.eigvals*t)*sum_i)
            sum_j += sum_k*j
        return sum_j
