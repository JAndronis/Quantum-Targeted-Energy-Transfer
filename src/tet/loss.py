import tensorflow as tf
assert tf.__version__ >= "2.0"
from math import factorial
from itertools import product
import numpy as np

DTYPE = tf.float32

class Loss:
    def __init__(self, const):
        self.coupling_lambda = tf.constant(const['coupling'], dtype=DTYPE)
        self.omegaA = tf.constant(const['omegaA'], dtype=DTYPE)
        self.omegaD = tf.constant(const['omegaD'], dtype=DTYPE)
        self.max_N = tf.constant(const['max_N'], dtype=DTYPE)
        self.max_t = tf.constant(const['max_t'], dtype=tf.int32)
        self.dim = const['max_N']+1

        initial_state = tf.TensorArray(DTYPE, size=self.dim)
        for n in range(self.dim):
            if n<self.dim-1: 
                initial_state = initial_state.write(n, tf.constant(0, dtype=DTYPE))
            else:
                initial_state = initial_state.write(n, self.max_N)
        self.initial_state = initial_state.stack()
        self.initial_state = tf.divide(self.initial_state, tf.linalg.norm(self.initial_state))

    def __call__(self, xA, xD):
        return self.loss(xA, xD)

    def createHamiltonian(self, xA, xD):
        h = tf.zeros((self.dim, self.dim), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(self.dim):
            n = tf.cast(i, dtype=DTYPE)
            for j in range(self.dim):
                if i==j:
                    diag_indices.append([i,j])
                    diag_updates.append(self.omegaD * n + 0.5 * xD * n ** 2\
                            + self.omegaA * (self.max_N - n) + 0.5 * xA * (self.max_N - n) ** 2)
                if i==j-1:
                    lower_diag_indices.append([i,j])
                    lower_diag_updates.append(-self.coupling_lambda * tf.sqrt((n + 1) * (self.max_N - n)))
                if i==j+1:
                    upper_diag_indices.append([i,j])
                    upper_diag_updates.append(-self.coupling_lambda * tf.sqrt(n * (self.max_N - n + 1)))

        h = tf.tensor_scatter_nd_update(h, diag_indices, diag_updates)
        h = tf.tensor_scatter_nd_update(h, upper_diag_indices, upper_diag_updates)
        h = tf.tensor_scatter_nd_update(h, lower_diag_indices, lower_diag_updates)
        return h
    
    def coeffs(self, xA, xD):
        problemHamiltonian = self.createHamiltonian(xA, xD)
        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)
        
        eigvecs = tf.cast(eigvecs, dtype=DTYPE)
        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(eigvecs[:,i], self.initial_state, 1))
        
        coeff_c = coeff_c.stack()
        coeff_b = eigvecs
        return coeff_c, coeff_b, eigvals
    
    def computeAverage(self, c, b, e):
        _time = self.max_t+1
        n = tf.TensorArray(DTYPE, size=_time)
        for t in range(_time):
            _t = tf.cast(t, dtype=tf.complex64)
            sum_j = tf.cast(0, dtype=tf.complex64)
            for j in range(self.dim):
                temp_b = tf.cast(b[j,:], dtype=tf.complex64)
                temp_c = tf.cast(c, dtype=tf.complex64)
                temp_e = tf.cast(e, dtype=tf.complex64)
                sum_i = tf.reduce_sum(temp_c*temp_b*tf.exp(-tf.complex(0.0,1.0)*temp_e*_t), 0)
                sum_k = tf.reduce_sum(temp_c*temp_b*tf.exp(tf.complex(0.0,1.0)*temp_e*_t)*sum_i, 0)
                j = tf.cast(j, dtype=tf.complex64)
                sum_j = sum_j+sum_k*j
            sum_j = tf.math.real(sum_j)
            n = n.write(t, sum_j)
        return n.stack()
        
    def loss(self, xA, xD):
        coeff_c, coeff_b, vals = self.coeffs(xA, xD)
        avg_N_list = self.computeAverage(coeff_c, coeff_b, vals)
        avg_N = tf.math.reduce_min(avg_N_list, name='Average_N')
        return avg_N

class LossMultiSite:
    def __init__(self, n, t, coupling_lambda, sites, omegas):
        self.max_N = tf.constant(n, dtype=DTYPE)
        self.max_N_np = n
        self.max_t = tf.constant(t, dtype=tf.int32)
        self.coupling_lambda = tf.constant(coupling_lambda, dtype=DTYPE)
        self.sites = sites
        self.omegas = tf.constant(omegas, dtype=DTYPE)

        self.dim = int( factorial(n+sites-1)/( factorial(n)*factorial(sites-1) ) )
        self.CombinationsBosons = self.derive()
        self.StatesDictionary = dict(zip(np.arange(self.dim, dtype=int), self.CombinationsBosons))

        #Assign initially all the bosons at the donor
        self.InitialState = self.StatesDictionary[0]
        self.InitialState = dict.fromkeys(self.InitialState, 0)
    
        self.InitialState['x0'] = self.max_N
        #Find the index
        self.InitialStateIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(self.InitialState)]
        self.Identity = np.identity(n=self.dim)
        self.InitialState = self.Identity[self.InitialStateIndex]
        self.initialState = tf.convert_to_tensor(self.InitialState, dtype=DTYPE)

    def __call__(self, site, xA, xD):
        if self.sites>2:
            zeros_list = [tf.constant(0, dtype=DTYPE) for _ in range(self.sites-1)]
            self.chis = [xD] + zeros_list + [xA]
        else: self.chis = [xD, xA]
        self.chis = [xD, 0, xA]
        self.targetState = site
        return self.loss()

    def getCombinations(self):
        return self.CombinationsBosons

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

    #Find the Hnm element of the Hamiltonian
    def ConstructElement(self, n, m):
        #First Term. Contributions due to the kronecker(n,m) elements
        Term1 = tf.constant(0, dtype=DTYPE)
        #Second Term. Various contributions
        Term2a, Term2b = tf.constant(0, dtype=DTYPE), tf.constant(0, dtype=DTYPE)
        
        if n==m:
            for k in range(self.sites):
                Term1 += self.omegas[k] * self.StatesDictionary[m]["x{}".format(k)] +\
                    0.5 * self.chis[k] * (self.StatesDictionary[m]["x{}".format(k)])**2
        else:
            for k in range(self.sites-1):
                #Find the number of bosons
                nk = self.StatesDictionary[m]["x{}".format(k)]
                nkplusone = self.StatesDictionary[m]["x{}".format(k+1)]
                _maxN = tf.get_static_value(self.max_N)

                #Term 2a/Important check
                if (nkplusone != _maxN) and (nk != 0): 
                    m1TildaState = self.StatesDictionary[m].copy()
                    m1TildaState["x{}".format(k)] = nk-1
                    m1TildaState["x{}".format(k+1)] = nkplusone+1

                    m1TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m1TildaState)]
                
                    if m1TildaIndex == n: Term2a += -self.coupling_lambda*np.sqrt((nkplusone+1)*nk)
                
                #Term 2b/Important check
                if (nkplusone != 0) and (nk != _maxN): 
                    #Find the new state/vol2
                    m2TildaState = self.StatesDictionary[m].copy()
                    m2TildaState["x{}".format(k)] = nk+1
                    m2TildaState["x{}".format(k+1)] = nkplusone-1

                    m2TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m2TildaState)]

                    if m2TildaIndex == n: Term2b += -self.coupling_lambda*np.sqrt(nkplusone*(nk+1))
                
        return Term1 + Term2a + Term2b

    def createHamiltonian(self):
        h = tf.TensorArray(dtype=DTYPE, size=self.dim*self.dim)
        for n in range(self.dim):
            for m in range(self.dim):
                h = h.write(n*self.dim+m, self.ConstructElement(n, m))
        h = h.stack()
        h = tf.reshape(h, shape=[self.dim, self.dim])
        return h

    def setCoeffs(self):
        problemHamiltonian = self.createHamiltonian()
        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)

        self.eigvals = tf.cast(eigvals, dtype=DTYPE)
        eigvecs = tf.cast(eigvecs, dtype=DTYPE)

        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(tf.cast(eigvecs[:,i], dtype=DTYPE), tf.cast(self.InitialState, dtype=DTYPE), 1))
        
        coeff_c = coeff_c.stack()
        coeff_b = eigvecs
        self.ccoeffs = coeff_c
        self.bcoeffs = coeff_b

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

    def loss(self):
        Data = tf.TensorArray(DTYPE, size=self.max_t+1)
        self.setCoeffs()
        for t in range(self.max_t+1):
            #print('\r t = {}'.format(t),end="")
            x = self._computeAverageCalculation(t)
            Data = Data.write(t, value=x)
        Data = Data.stack()
        return tf.reduce_min(Data)
    
if __name__=="__main__":
    loss = LossMultiSite(3, 100, 0.1, 3, [-3,3,3])
    @tf.function
    def test():
        n = loss(site='x0', xD=0.5, xA=1)
        return n
    print(test())