import os
import tensorflow as tf
from math import factorial
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
assert tf.__version__ >= "2.0"

DTYPE = tf.float64


class Loss:
    """
    A class designed for computing the loss function

    Args:
        const (dict):  Refer to the system_constants dictionary in constants.py.
    """

    def __init__(self, const):
        # Import the parameters of the problem
        self.bcoeffs = None
        self.ccoeffs = None
        self.initial_state = None
        self.eigvals = None
        self.max_N = tf.constant(const['max_N'], dtype=DTYPE)
        self.NpointsT = const['timesteps']
        self.max_N_np = const['max_N']
        self.max_t = tf.constant(const['max_t'], dtype=tf.int32)
        self.coupling_lambda = tf.constant(const['coupling'], dtype=DTYPE)
        self.sites = const['sites']
        self.omegas = tf.constant(const['omegas'], dtype=DTYPE)

        # Default values for chis and target site, will be replaced by __call__()
        self.chis = const['chis']
        self.targetState = const['sites'] - 1

        # Define some other helpful variables
        self.dim = int(factorial(const['max_N'] + const['sites'] - 1) / (
                factorial(const['max_N']) * factorial(const['sites'] - 1)))

        # Initialize states array
        self.states = np.zeros((self.dim, self.sites))
        self.T = np.zeros(self.dim)
        self.sorted_indices = np.empty(self.T.shape)
        self.derive()
        self.getHashArray()

    def __call__(self, chis, site=0, single_value=True) -> tf.Tensor:
        self.chis = chis
        try:
            if type(site) != int:
                raise ValueError
            self.targetState = site
        except ValueError:
            if type(site) == str:
                self.targetState = int(site[-1])
            else:
                raise ValueError("Invalid type for site variable. Must be int.")
        return self.loss(single_value)

    def derive(self):
        """
        A function that generates all the possible configurations of distributing N indistinguishable bosons in f distinguishable sites.
        """

        # Initially, all the bosons belong to the donor site.
        self.states[0, 0] = tf.get_static_value(self.max_N)

        v, k = 0, 0
        while v < self.dim - 1:

            for i in range(k):
                self.states[v + 1, i] = self.states[v, i]

            self.states[v + 1, k] = self.states[v, k] - 1
            s = 0
            for i in range(k + 1):
                s += self.states[(v + 1), i]
            self.states[v + 1, k + 1] = tf.get_static_value(self.max_N) - s

            for j in range(k + 2, self.sites):
                self.states[v + 1, j] = 0

            _k = 0
            condition = True
            while _k < self.sites - 1:
                _i = _k + 1
                if _i >= self.sites - 1:
                    _i = self.sites - 1
                    if self.states[v + 1, _i] != 0:
                        condition = False
                    else:
                        condition = True
                else:
                    while _i < self.sites - 1:
                        if self.states[v + 1, _i] != 0:
                            condition = False
                            break
                        else:
                            condition = True
                        _i += 1

                if not condition:
                    _k += 1
                else:
                    break
            if condition:
                k = _k
            v += 1

    def getHash(self, state):
        s = 0
        for i in range(self.sites):
            s += np.sqrt(100 * (i + 1) + 3) * state[i]
        return s

    def getHashArray(self):
        for i in range(self.dim):
            self.T[i] = self.getHash(self.states[i])
        self.sorted_indices = np.argsort(self.T)

    # Deduce the Hnm element of the Hamiltonian
    def ConstructElement(self, n, m):

        # * First Term. Contributions due to the Kronecker(n,m) elements
        term1 = tf.constant(0, dtype=DTYPE)
        # * Second Term. Various contributions
        term2a, term2b = tf.constant(0, dtype=DTYPE), tf.constant(0, dtype=DTYPE)

        if n == m:
            for k in range(self.sites):
                term1 += self.omegas[k] * self.states[m, k] + \
                         0.5 * self.chis[k] * (self.states[m, k]) ** 2
        else:
            for k in range(self.sites - 1):
                # Find the number of bosons
                nk = self.states[m, k]
                nkplusone = self.states[m, k + 1]
                _maxN = tf.get_static_value(self.max_N)

                # Term 2a/Important check
                if (nkplusone != _maxN) and (nk != 0):
                    m1_tilda_state = self.states[m].copy()
                    m1_tilda_state[k] = nk - 1
                    m1_tilda_state[k + 1] = nkplusone + 1
                    state_hash = self.getHash(m1_tilda_state)
                    _idx = self.sorted_indices[np.searchsorted(self.T, state_hash, sorter=self.sorted_indices)]

                    if _idx == n:
                        term2a -= self.coupling_lambda * np.sqrt((nkplusone + 1) * nk)

                # Term 2b/Important check
                if (nkplusone != 0) and (nk != _maxN):
                    # Find the new state/vol2
                    m2_tilda_state = self.states[m].copy()
                    m2_tilda_state[k] = nk + 1
                    m2_tilda_state[k + 1] = nkplusone - 1
                    state_hash = self.getHash(m2_tilda_state)
                    _idx = self.sorted_indices[np.searchsorted(self.T, state_hash, sorter=self.sorted_indices)]

                    if _idx == n:
                        term2b -= self.coupling_lambda * np.sqrt(nkplusone * (nk + 1))

        return term1 + term2a + term2b

    # ! Constructing the Hamiltonian operator.
    def createHamiltonian(self):

        h = tf.TensorArray(dtype=DTYPE, size=self.dim * self.dim)
        for n in range(self.dim):
            for m in range(self.dim):
                h = h.write(n * self.dim + m, self.ConstructElement(n, m))
        h = h.stack()
        h = tf.reshape(h, shape=[self.dim, self.dim])
        return h

    # ! Given a set of nonlinearity parameters, compute the coefficients needed according to PRL.
    def setCoefs(self):
        problem_hamiltonian = self.createHamiltonian()

        eigvals, eigvecs = tf.linalg.eigh(problem_hamiltonian)
        self.eigvals = tf.cast(eigvals, dtype=DTYPE)
        eigvecs = tf.cast(eigvecs, dtype=DTYPE)

        self.initial_state = np.zeros(self.sites)
        self.initial_state[0] = tf.get_static_value(self.max_N)
        state_hash = self.getHash(self.initial_state)
        init_idx = np.searchsorted(self.T, state_hash, sorter=self.sorted_indices)
        self.initial_state = np.identity(self.dim)[init_idx]
        self.initial_state = tf.convert_to_tensor(self.initial_state, dtype=DTYPE)

        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(tf.cast(eigvecs[:, i], dtype=DTYPE),
                                                    tf.cast(self.initial_state, dtype=DTYPE), 1))

        coeff_c = coeff_c.stack()

        self.ccoeffs = coeff_c
        self.bcoeffs = eigvecs

    # ! Computing the loss function given the time step .
    def _computeAverageCalculation(self, t):
        sum_j = tf.TensorArray(dtype=tf.complex64, size=self.dim)
        for j in range(self.dim):
            c = tf.cast(self.ccoeffs, dtype=tf.complex64)
            b = tf.cast(self.bcoeffs, dtype=tf.complex64)
            e = tf.cast(self.eigvals, dtype=tf.complex64)
            _t = tf.cast(t, dtype=tf.complex64)
            sum_i = tf.reduce_sum(c * b[j, :] * tf.exp(tf.complex(0., -1.) * e * _t))
            sum_k = tf.reduce_sum(tf.math.conj(c) * tf.math.conj(b[j, :]) * tf.exp(tf.complex(0., 1.) * e * _t) * sum_i)
            sum_j = sum_j.write(j, value=sum_k * self.states[j][self.targetState])
        sum_j = tf.reduce_sum(sum_j.stack())
        return tf.cast(tf.math.real(sum_j), dtype=DTYPE)

    # ! Computing the loss function given a Hamiltonian corresponding to one combination of nonlinearity parameters
    def loss(self, single_value=True):
        data = tf.TensorArray(DTYPE, size=self.NpointsT)
        self.setCoefs()
        t_span = np.linspace(0, tf.get_static_value(self.max_t), self.NpointsT)
        for index_t, t in enumerate(t_span):
            # print('\r t = {}'.format(t),end="")
            x = self._computeAverageCalculation(t)
            data = data.write(index_t, value=x)
        data = data.stack()
        if single_value:
            if self.targetState == self.sites - 1:
                return self.max_N - tf.reduce_max(data)
            else:
                return tf.reduce_min(data)
        else:
            return data
