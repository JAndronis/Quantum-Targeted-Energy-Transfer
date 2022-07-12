import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
assert tf.__version__ >= "2.0"
from math import factorial
from itertools import product
import numpy as np
from constants import TensorflowParams

DTYPE = TensorflowParams['DTYPE']

class Loss:
    """
    Class Loss: A class designed for computing the loss function
    Documentation:
        * const:  Refer to the constants dictionary in constants.py.
    """
    def __init__(self, const):
        #! Import the parameters of the problem
        self.max_N = tf.constant(const['max_N'], dtype=DTYPE)
        self.NpointsT = const['timesteps']
        self.max_N_np = const['max_N']
        self.max_t = tf.constant(const['max_t'], dtype=tf.int32)
        self.coupling_lambda = tf.constant(const['coupling'], dtype=DTYPE)
        self.sites = const['sites']
        self.omegas = tf.constant(const['omegas'], dtype=DTYPE)

        # default values for chis and target site, will be replaced by __call__()
        self.chis = const['chis']
        self.targetState = const['sites'] - 1
        
        #! Define some other helpful variables
        self.dim = int( factorial(const['max_N']+const['sites']-1)/( factorial(const['max_N'])*factorial(const['sites']-1) ) )


    def __call__(self, *args, site=int(), single_value=True):
        self.chis = list(args)
        self.targetState = site
        return self.loss(single_value)

    def getCombinations(self):
        return self.CombinationsBosons

    #! Computing the Fock states
    def derive(self):

        self.states = np.zeros((self.dim, self.sites))
        self.states[0, 0] = self.max_N

        v = 0
        k = 0
        while v < self.dim - 1:

            for i in range(0, k): self.states[v+1, i] = self.states[v, i]

            self.states[v+1, k] = self.states[v, k] - 1
            self.states[v+1, k+1] = self.max_N - self.states[v+1, 0:k+2].sum()

            for j in range(k+2, self.sites): self.states[v+1, j] = 0

            _k = 0
            condition = True
            while _k < self.sites - 1:
                _i = _k+1
                if _i >= self.sites - 1:
                    _i = self.sites - 1
                    if self.states[v+1, _i] != 0: condition = False
                    else: condition = True
                else:
                    while _i < self.sites - 1:
                        if self.states[v+1, _i] != 0:
                            condition = False
                            break
                        else: condition = True
                        _i += 1
                
                if not condition: _k += 1
                else: break
            if condition: k = _k
            v += 1

    def getHash(self, state):
        s = 0
        for i in range(self.sites):
            s += np.sqrt(100 * (i + 1) + 3) * state[i]
        return s

    def getHashArray(self):
        self.T = np.zeros(self.dim)
        for i in range(self.dim):
            self.T[i] = self.getHash(self.states[i])
        self.sorted_indeces = np.argsort(self.T)

    #! Deduce the Hnm element of the Hamiltonian
    def ConstructElement(self, n, m):
        
        #* First Term. Contributions due to the Kronecker(n,m) elements
        Term1 = tf.constant(0, dtype=DTYPE)
        #* Second Term. Various contributions
        Term2a, Term2b = tf.constant(0, dtype=DTYPE), tf.constant(0, dtype=DTYPE)
        
        if n==m:
            for k in range(self.sites):
                Term1 += self.omegas[k] * self.states[m, k] +\
                    0.5 * self.chis[k] * (self.states[m, k])**2
        else:
            for k in range(self.sites-1):
                #Find the number of bosons
                nk = self.states[m, k]
                nkplusone = self.states[m, k+1]
                _maxN = tf.get_static_value(self.max_N)

                #Term 2a/Important check
                if (nkplusone != _maxN) and (nk != 0): 
                    m1TildaState = self.states[m].copy()
                    m1TildaState[k] = nk - 1
                    m1TildaState[k+1] = nkplusone + 1
                    state_hash = self.getHash(m1TildaState)
                    _idx = self.sorted_indeces[np.searchsorted(self.T, state_hash, sorter=self.sorted_indeces)]
                
                    if _idx == n: Term2a -= self.coupling_lambda*np.sqrt((nkplusone+1)*nk)
                
                #Term 2b/Important check
                if (nkplusone != 0) and (nk != _maxN):
                    #Find the new state/vol2
                    m2TildaState = self.states[m].copy()
                    m2TildaState[k] = nk + 1
                    m2TildaState[k+1] = nkplusone - 1
                    state_hash = self.getHash(m2TildaState)
                    _idx = self.sorted_indeces[np.searchsorted(self.T, state_hash, sorter=self.sorted_indeces)]

                    if _idx == n: Term2b -= self.coupling_lambda*np.sqrt(nkplusone*(nk+1))
                
        return Term1 + Term2a + Term2b

    #! Constructing the Hamiltonian operator.
    def createHamiltonian(self):

        self.derive()
        self.getHashArray()
        
        h = tf.TensorArray(dtype=DTYPE, size=self.dim*self.dim)
        for n in range(self.dim):
            for m in range(self.dim):
                h = h.write(n*self.dim+m, self.ConstructElement(n, m))
        h = h.stack()
        h = tf.reshape(h, shape=[self.dim, self.dim])
        return h

    #! Given a set of non linearity parameters, compute the coefficients needed according to PRL.
    def setCoeffs(self):
        problemHamiltonian = self.createHamiltonian()

        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)
        self.eigvals = tf.cast(eigvals, dtype=DTYPE)
        eigvecs = tf.cast(eigvecs, dtype=DTYPE)

        self.InitialState = np.zeros(self.sites)
        self.InitialState[0] = self.max_N
        state_hash = self.getHash(self.InitialState)
        init_idx = np.searchsorted(self.T, state_hash, sorter=self.sorted_indeces)
        self.InitialState = np.identity(self.dim)[init_idx]
        self.InitialState = tf.convert_to_tensor(self.InitialState, dtype=DTYPE)

        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(tf.cast(eigvecs[:,i], dtype=DTYPE), tf.cast(self.InitialState, dtype=DTYPE), 1))
        
        coeff_c = coeff_c.stack()
    
        self.ccoeffs = coeff_c
        self.bcoeffs = eigvecs

    #! Computing the loss function given the time step .
    def _computeAverageCalculation(self, t):
        sum_j = tf.TensorArray(dtype=tf.complex64, size=self.dim)
        for j in range(self.dim):
            c = tf.cast(self.ccoeffs, dtype=tf.complex64)
            b = tf.cast(self.bcoeffs, dtype=tf.complex64)
            e = tf.cast(self.eigvals, dtype=tf.complex64)
            _t = tf.cast(t, dtype=tf.complex64)
            sum_i = tf.reduce_sum(c*b[j,:]*tf.exp(tf.complex(0.,-1.)*e*_t))
            sum_k = tf.reduce_sum(tf.math.conj(c)*tf.math.conj(b[j,:])*tf.exp(tf.complex(0.,1.)*e*_t)*sum_i)
            sum_j = sum_j.write(j, value=sum_k*self.states[j][self.targetState])
        sum_j = tf.reduce_sum(sum_j.stack())
        return tf.math.real(sum_j)

    #! Computing the loss function given a Hamiltonian correspodning to one combination of non linearity parameters
    def loss(self, single_value=True):
        Data = tf.TensorArray(DTYPE, size=self.NpointsT)
        self.setCoeffs()
        t_span = np.linspace(0,self.max_t,self.NpointsT)
        for indext,t in enumerate(t_span):
            #print('\r t = {}'.format(t),end="")
            x = self._computeAverageCalculation(t)
            Data = Data.write(indext, value=x)
        Data = Data.stack()
        if single_value:
            if not self.targetState==f'x{self.sites}':
                return tf.reduce_min(Data)
            else:
                return self.max_N - tf.reduce_max(Data)
        else: return Data

def main(maxn, sites):
    import constants as const
    import matplotlib.pyplot as plt

    c = const.constants
    c['max_N'] = maxn
    c['sites'] = sites
    c['omegas'] = [3 for _ in range(c['sites'])]
    c['omegas'][0] = - c['omegas'][0]
    chi = (c['omegas'][-1] - c['omegas'][0]) / c['max_N']
    chi_m = -(chi) - 1
    c['chis'] = [chi] + [chi_m for _ in range(c['sites']-2)] + [chi]
    c['timesteps'] = 50
    l = Loss(c)
    h = l.createHamiltonian().numpy()
    # h_bool = np.zeros(h.shape)
    # for i in range(h.shape[0]):
    #     for j in range(h.shape[1]):
    #         if h[i,j] != 0:
    #             h_bool[i,j] = 1
    #         else:
    #             h_bool[i,j] = 0
    # plt.imshow(h_bool, cmap='gray_r')
    # plt.show()
    # print(c['max_N'] - max(l(*c["chis"], site=const.constants['sites']-1, single_value=False).numpy()))