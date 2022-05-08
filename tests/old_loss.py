import tensorflow as tf
assert tf.__version__ >= "2.0"
from math import factorial
from itertools import product
import numpy as np

DTYPE = tf.float32

class Loss:
    """
    Class that contains functions to compute the average number of bosons on the donor site in a dimer system.
    """
    def __init__(self, const):

        # Initialize the parameters of the problem from the provided dict.
        self.coupling_lambda = tf.constant(const['coupling'], dtype=DTYPE)
        self.omegaA = tf.constant(const['omegaA'], dtype=DTYPE)
        self.omegaD = tf.constant(const['omegaD'], dtype=DTYPE)
        self.max_N = tf.constant(const['max_N'], dtype=DTYPE)
        self.max_t = tf.constant(const['max_t'], dtype=tf.int32)
        self.dim = const['max_N']+1

        # Initialize the initial state of the dimer.
        initial_state = tf.TensorArray(DTYPE, size=self.dim)
        for n in range(self.dim):
            if n<self.dim-1: 
                initial_state = initial_state.write(n, tf.constant(0, dtype=DTYPE))
            else:
                initial_state = initial_state.write(n, tf.constant(1, dtype=DTYPE))
        self.initial_state = initial_state.stack()

    def __call__(self, xA, xD):
        return self.loss(xA, xD)

    def createHamiltonian(self, xA, xD):
        """
        Function to create a system's Hamiltonian. 

        Args:
            * xA (tf.Variable()): The non linearity parameter of the acceptor site.
            * xD (tf.Variable()): The non linearity parameter of the donor site.

        Returns:
            * tf.Tensor(): The Hamiltonian of the system in a tf.Tensor format.
        """

        h = tf.zeros((self.dim, self.dim), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(self.dim):
            n = tf.cast(i, dtype=DTYPE) # i bosons at the donor
            for j in range(self.dim):

                # Term coming from the two independent Hamiltonians
                if i==j:
                    diag_indices.append([i,j])
                    diag_updates.append(self.omegaD * n + 0.5 * xD * n ** 2\
                            + self.omegaA * (self.max_N - n) + 0.5 * xA * (self.max_N - n) ** 2)
                
                # First term from interaction
                if i==j-1:
                    lower_diag_indices.append([i,j])
                    lower_diag_updates.append(-self.coupling_lambda * tf.sqrt((n + 1) * (self.max_N - n)))
                
                # Second term from interaction
                if i==j+1:
                    upper_diag_indices.append([i,j])
                    upper_diag_updates.append(-self.coupling_lambda * tf.sqrt(n * (self.max_N - n + 1)))

        # Assign values to correct indeces of the tensor
        h = tf.tensor_scatter_nd_update(h, diag_indices, diag_updates)
        h = tf.tensor_scatter_nd_update(h, upper_diag_indices, upper_diag_updates)
        h = tf.tensor_scatter_nd_update(h, lower_diag_indices, lower_diag_updates)
        return h
    
    def coeffs(self, xA, xD):
        """
        Function that calculates the probability coefficients for the basis states and eigenstates.

        Args:
            * xA (tf.Variable()): The non linearity parameter of the acceptor site.
            * xD (tf.Variable()): The non linearity parameter of the donor site.

        Returns:
            * tf.Tensor(), tf.Tensor(), tf.Tensor(): Tuple of the required values as tf.Tensors 
                                                    for the average number of bosons computation.
                                                    The first value represents the C coefficients, 
                                                    the second the b coefficients and the last one,
                                                    the eigenvalues of the Hamiltonian.
        """

        # Compute the Hamiltonian and diagonalize it to find its eigenstates and eigenvalues.
        problemHamiltonian = self.createHamiltonian(xA, xD)
        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)
        
        # Cast variables to correct DTYPE for following operations
        eigvecs = tf.cast(eigvecs, dtype=DTYPE)

        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(eigvecs[:,i], self.initial_state, 1))
        
        coeff_c = coeff_c.stack()
        coeff_b = eigvecs
        return coeff_c, coeff_b, eigvals
    
    def computeAverage(self, c, b, e):
        """
        Function that computes the time evolution of the average number of bosons observable.

        Args:
            * c (tf.Tensor()): tf.Tensor containing the values of the C probability coefficient.
            * b (tf.Tensor()): tf.Tensor containing the values of the b probability coefficient.
            * e (tf.Tensor()): tf.Tensor containing the eigenvalues of the Hamiltonian.

        Returns:
            * tf.Tensor(): The average number of bosons after time max_t as defined in the class initialization.
                           Returns a tf.Tensor().
        """
        
        _time = self.max_t+1
        n = tf.TensorArray(DTYPE, size=_time)
        for t in range(_time):
            _t = tf.cast(t, dtype=tf.complex64) # cast a helper variable at the current timestep for complex calculations
            sum_j = tf.cast(0, dtype=tf.complex64)
            for j in range(self.dim):

                # initialize helper variables as the correct type.
                temp_b = tf.cast(b[j,:], dtype=tf.complex64)
                temp_c = tf.cast(c, dtype=tf.complex64)
                temp_e = tf.cast(e, dtype=tf.complex64)

                # compute the sum
                sum_i = tf.reduce_sum(temp_c*temp_b*tf.exp(-tf.complex(0.0,1.0)*temp_e*_t), 0)
                sum_k = tf.reduce_sum(temp_c*temp_b*tf.exp(tf.complex(0.0,1.0)*temp_e*_t)*sum_i, 0)
                j = tf.cast(j, dtype=tf.complex64)
                sum_j = sum_j+sum_k*j
            sum_j = tf.math.real(sum_j)
            n = n.write(t, sum_j)
        return n.stack()
        
    def loss(self, xA, xD):
        """
        Function that calls the necessery functions to compute the minimum value of the time evolved observable N.

        Args:
            * xA (tf.Variable()): The non linearity parameter of the acceptor site.
            * xD (tf.Variable()): The non linearity parameter of the donor site.

        Returns:
            * tf.Tensor(): tf.Tensor with shape=() containing only a single value of the minimum number of bosons on the donor
                           after time evolution of the observable for time max_t, defined in __init__.
        """
        coeff_c, coeff_b, vals = self.coeffs(xA, xD)
        avg_N_list = self.computeAverage(coeff_c, coeff_b, vals)
        avg_N = tf.math.reduce_min(avg_N_list, name='Average_N')
        return avg_N