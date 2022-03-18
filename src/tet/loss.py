import tensorflow as tf
assert tf.__version__ >= "2.0"

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
    