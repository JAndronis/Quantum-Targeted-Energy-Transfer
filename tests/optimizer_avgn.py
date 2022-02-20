import sys
assert sys.version_info >= (3,6)
import os

import numpy as np
import matplotlib.pyplot as plt
import time

from tet.data_process import createDir
from tet.saveFig import saveFig

import tensorflow as tf
assert tf.__version__ >= "2.0"

# enable memory growth
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

# constants
DTYPE = tf.complex64
CHIA = tf.constant(0, dtype=tf.float32)
CHID = tf.constant(0, dtype=tf.float32)
LAMBDA = tf.constant(1, dtype=tf.float32)
OMEGA_A = tf.constant(-3, dtype=tf.float32)
OMEGA_D = tf.constant(3, dtype=tf.float32)
MAX_N = tf.constant(4, dtype=tf.float32)
MAX_T = tf.constant(25, dtype=tf.float32)

class Loss:
    def __init__(self):
        self.chiD = CHIA
        self.chiA = CHID
        self.coupling_lambda = LAMBDA
        self.omegaA = OMEGA_A
        self.omegaD = OMEGA_D
        self.max_N = MAX_N
        self.max_t = MAX_T
        
        dim = tf.cast(self.max_N, dtype=tf.int32)
        self.initial_state = tf.zeros(dim.numpy()+1,dtype=DTYPE)
        initial_indices = []
        initial_updates = []
        for n in range(dim.numpy()+1):
            initial_indices.append([n])
            i = tf.cast(n, dtype=tf.float32)
            initial_updates.append(tf.exp(-(self.max_N-i)**2))
        self.initial_state = tf.tensor_scatter_nd_update(self.initial_state, initial_indices, initial_updates)
        self.initial_state = self.initial_state / tf.linalg.norm(self.initial_state)

    def __call__(self, xA, xD):
        return self.loss(xA, xD)

    def createHamiltonian(self, chiA, chiD):
        dim = tf.cast(self.max_N, dtype=tf.int32)
        h = tf.zeros((dim.numpy()+1, dim.numpy()+1), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(dim.numpy() + 1):
            n = tf.cast(i, dtype=tf.float32)
            for j in range(dim.numpy() + 1):
                if i==j: 
                    diag_indices.append([i,j])
                    diag_updates.append(self.omegaD * n + 0.5 * chiD * n ** 2\
                            + self.omegaA * (self.max_N - n) + 0.5 * chiA * (self.max_N - n) ** 2)
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
        self.eigvals, self.eigvecs = tf.linalg.eigh(problemHamiltonian)
        
        dim = tf.cast(self.max_N, dtype=tf.int32)
        self.coeff_c = tf.zeros(dim.numpy()+1, dtype=tf.complex64)
        c_indices = []
        c_updates = []
        for i in range(dim.numpy()+1): 
            c_indices.append([i])
            c_updates.append(tf.tensordot(self.eigvecs[:,i], self.initial_state, 1))
        
        self.coeff_c = tf.tensor_scatter_nd_update(self.coeff_c, c_indices, c_updates)
        self.coeff_b = self.eigvecs
    
    def computeAverage(self):
        avg_N = []
        _time = range(0, tf.cast(self.max_t, dtype=tf.int32).numpy()+1)

        for t in _time:
            avg_N.append(self._computeAverageCalculation(t))
        return tf.math.real(avg_N)

    def _computeAverageCalculation(self, t):
        sum_j = 0
        dim = tf.cast(self.max_N, dtype=tf.int32)
        for j in range(dim.numpy()+1):
            sum_i = tf.add_n(self.coeff_c*self.coeff_b[j,:]*tf.exp(-tf.complex(0.0,1.0)*self.eigvals*t))
            sum_k = tf.add_n(self.coeff_c*self.coeff_b[j,:]*tf.exp(tf.complex(0.0,1.0)*self.eigvals*t)*sum_i)
            sum_j += sum_k*j
        return sum_j
    
    def loss(self, xA, xD):
        self.coeffs(xA, xD)
        avg_N = self.computeAverage()
        avg_N = tf.math.reduce_min(avg_N)
        return avg_N

def train():
    mylosses = []
    tol = 1e-8
    max_iter = 1000
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    xA = tf.Variable(initial_value=CHIA, trainable=True, dtype=tf.float32)
    xD = tf.Variable(initial_value=CHID, trainable=True, dtype=tf.float32)
    xA_best = tf.Variable(initial_value=0, dtype=tf.float32)
    xD_best = tf.Variable(initial_value=0, dtype=tf.float32)
    mylosses.append(4)
    best_loss = 4
    counter = 0
    d_data = []
    a_data = []
    loss = Loss()

    # return self.loss(tf.constant(1.549), tf.constant(-1.49))

    t0 = time.time()
    for epoch in range(max_iter):
        xA_init = xA.numpy()
        xD_init = xD.numpy()

        with tf.GradientTape() as t:
            t.watch([xA, xD])
            current_loss = loss(xA, xD)
        # _train = opt.minimize(current_loss, var_list=(xA, xD), tape=t)
        grads = t.gradient(current_loss, [xA, xD])
        opt.apply_gradients(zip(grads, [xA, xD]))
        
        if epoch%10 ==0: print(f'Loss:{current_loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        # print(f'Loss:{current_loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        
        errorA = np.abs(xA.numpy() - xA_init)
        errorD = np.abs(xD.numpy() - xD_init)

        mylosses.append(abs(current_loss.numpy()))
        if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
            xA_best.assign(xA.numpy())
            xD_best.assign(xD.numpy())
            best_loss = mylosses[epoch+1]

        counter += 1
        if counter%10 == 0:
            d_data.append(xD.numpy())
            a_data.append(xA.numpy())

        if np.abs(current_loss.numpy()) < tol:
            del t
            break
        
        if errorA < tol:
            del t
            print('Stopped training because of xA_new-xA_old =', errorA)
            break

        if errorD < tol:
            del t
            print('Stopped training because of xD_new-xD_old =', errorA)
            break
        
    t1 = time.time()
    dt = t1-t0

    print("\nApproximate value of chiA:", xA_best.numpy(), 
                "\nApproximate value of chiD:", xD_best.numpy(),
                "\nLoss:", best_loss,
                "\nOptimizer Iterations:", opt.iterations.numpy(), 
                "\nTraining Time:", dt,
                "\n"+40*"-")

    plt.plot(mylosses)
    plt.show()

    # return xA_best.numpy(), xD_best.numpy(), best_loss, a_data, d_data
    return xA_best.numpy(), xD_best.numpy()

if __name__=="__main__":
    train()