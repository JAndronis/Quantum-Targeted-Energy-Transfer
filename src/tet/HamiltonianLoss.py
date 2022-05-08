import tensorflow as tf
assert tf.__version__ >= "2.0"
from math import factorial
from itertools import product
import numpy as np
from constants import TensorflowParams

DTYPE = TensorflowParams['DTYPE']

class Loss:
    def __init__(self, const):
        #! Import the parameters of the problem
        self.max_N = tf.constant(const['max_N'], dtype=DTYPE)
        self.max_N_np = const['max_N']
        self.max_t = tf.constant(const['max_t'], dtype=tf.int32)
        self.coupling_lambda = tf.constant(const['coupling'], dtype=DTYPE)
        self.sites = const['sites']
        self.omegas = tf.constant(const['omegas'], dtype=DTYPE)
        
        #! Define some other helpful variables
        self.dim = int( factorial(const['max_N']+const['sites']-1)/( factorial(const['max_N'])*factorial(const['sites']-1) ) )
        self.CombinationsBosons = self.derive()
        self.StatesDictionary = dict(zip(np.arange(self.dim, dtype=int), self.CombinationsBosons))

        #! Define the initial state
        # Assign initially all the bosons at the donor
        self.InitialState = self.StatesDictionary[0]
        self.InitialState = dict.fromkeys(self.InitialState, 0)
        self.InitialState['x0'] = self.max_N

        # Find the index
        self.InitialStateIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(self.InitialState)]
        I = np.identity(n=self.dim)
        # Choose Fock state that matches the initial state
        self.InitialState = I[self.InitialStateIndex]   
        self.initialState = tf.convert_to_tensor(self.InitialState, dtype=DTYPE)


    def __call__(self, *args, site=str()):
        self.chis = list(args)
        self.targetState = site
        return self.loss()


    def getCombinations(self):
        return self.CombinationsBosons

    #! Computing the Fock states
    def derive(self):
        space = [np.arange(self.max_N_np+1) for _ in range(self.sites)]
        values = [i for i in product(*space) if sum(i)==self.max_N_np]
        keys = [f"x{i}" for i in range(self.sites)]
        solution = []
        kv = []
        for i in range(self.dim):
            temp = []
            for j in range(self.sites):
                temp.append([keys[j], values[i][j]])
            kv.append(temp)
            solution.append(dict(kv[i]))

        return solution

    #! Deduce the Hnm element of the Hamiltonian
    def ConstructElement(self, n, m):
        #* First Term. Contributions due to the Kronecker(n,m) elements
        Term1 = tf.constant(0, dtype=DTYPE)
        #* Second Term. Various contributions
        Term2a, Term2b = tf.constant(0, dtype=DTYPE), tf.constant(0, dtype=DTYPE)
        
        if n==m:
            for k in range(self.sites):
                Term1 += self.omegas[k] * self.StatesDictionary[m][f"x{k}"] +\
                    0.5 * self.chis[k] * (self.StatesDictionary[m][f"x{k}"])**2
        else:
            for k in range(self.sites-1):
                #Find the number of bosons
                nk = self.StatesDictionary[m][f"x{k}"]
                nkplusone = self.StatesDictionary[m][f"x{k+1}"]
                _maxN = tf.get_static_value(self.max_N)

                #Term 2a/Important check
                if (nkplusone != _maxN) and (nk != 0): 
                    m1TildaState = self.StatesDictionary[m].copy()
                    m1TildaState[f"x{k}"] = nk-1
                    m1TildaState[f"x{k+1}"] = nkplusone+1

                    m1TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m1TildaState)]
                
                    if m1TildaIndex == n: Term2a += -self.coupling_lambda*np.sqrt((nkplusone+1)*nk)
                
                #Term 2b/Important check
                if (nkplusone != 0) and (nk != _maxN): 
                    #Find the new state/vol2
                    m2TildaState = self.StatesDictionary[m].copy()
                    m2TildaState[f"x{k}"] = nk+1
                    m2TildaState[f"x{k+1}"] = nkplusone-1

                    m2TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m2TildaState)]

                    if m2TildaIndex == n: Term2b += -self.coupling_lambda*np.sqrt(nkplusone*(nk+1))
                
        return Term1 + Term2a + Term2b

    #! Constructing the Hamiltonian operator.
    def createHamiltonian(self):
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
            sum_j = sum_j.write(j, value=sum_k*self.StatesDictionary[j][self.targetState])
        sum_j = tf.reduce_sum(sum_j.stack())
        return tf.math.real(sum_j)

    #! Computing the loss function given a Hamiltonian correspodning to one combination of non linearity parameters
    def loss(self):
        Data = tf.TensorArray(DTYPE, size=self.max_t+1)
        self.setCoeffs()
        for t in range(self.max_t+1):
            #print('\r t = {}'.format(t),end="")
            x = self._computeAverageCalculation(t)
            Data = Data.write(t, value=x)
        Data = Data.stack()
        if not self.targetState==f'{list(self.StatesDictionary[0].keys())[-1]}':
            return tf.reduce_min(Data)
        else:
            return self.max_N - tf.reduce_max(Data)

