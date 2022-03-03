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
import keras.backend as K

# enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# constants
DTYPE = tf.float32
CHIA = tf.constant(1, dtype=DTYPE)
CHID = tf.constant(-3, dtype=DTYPE)
LAMBDA = tf.constant(0.1, dtype=DTYPE)
OMEGA_A = tf.constant(-3, dtype=DTYPE)
OMEGA_D = tf.constant(3, dtype=DTYPE)
MAX_N = tf.constant(4, dtype=DTYPE)
MAX_T = tf.constant(25, dtype=tf.int32)

OPT = tf.keras.optimizers.Adam(learning_rate=0.01)

class Loss:
    def __init__(self):
        self.coupling_lambda = LAMBDA
        self.omegaA = OMEGA_A
        self.omegaD = OMEGA_D
        self.max_N = MAX_N
        self.max_t = MAX_T
        self.dim = tf.cast(MAX_N+1, dtype=tf.int32)

        initial_state = tf.TensorArray(DTYPE, size=self.dim)
        for n in range(self.dim):
            initial_state = initial_state.write(n, tf.exp(-tf.pow((self.max_N-n), 2)))
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
        _time = MAX_T+1
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

@tf.function
def compute_loss(xA, xD):
    return Loss().loss(xA, xD)

def get_grads(xA, xD):
    with tf.GradientTape() as t:
            t.watch([xA, xD])
            loss = compute_loss(xA, xD)
    grads = t.gradient(loss, [xA, xD])
    del t
    return grads, loss
    
@tf.function
def apply_grads(xA, xD):
    grads, loss = get_grads(xA, xD)
    OPT.apply_gradients(zip(grads, [xA, xD]))
    return loss

def train(fig_path):
    mylosses = []
    tol = 1e-8
    max_iter = 1000
    xA = tf.Variable(initial_value=CHIA, trainable=True, dtype=tf.float32)
    xD = tf.Variable(initial_value=CHID, trainable=True, dtype=tf.float32)
    xA_best = tf.Variable(initial_value=0, dtype=tf.float32)
    xD_best = tf.Variable(initial_value=0, dtype=tf.float32)
    mylosses.append(3.9)
    best_loss = 4
    counter = 0
    d_data = []
    a_data = []
    loss = Loss()
    change = 0

    t0 = time.time()
    for epoch in range(max_iter):
        xA_init = xA.numpy()
        xD_init = xD.numpy()
        loss = apply_grads(xA, xD)
        if loss.numpy()<=1 and change==0: 
            K.set_value(OPT.learning_rate, 0.01)
            change += 1
        if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        
        errorA = np.abs(xA.numpy() - xA_init)
        errorD = np.abs(xD.numpy() - xD_init)

        mylosses.append(loss.numpy())
        if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
            xA_best.assign(xA.numpy())
            xD_best.assign(xD.numpy())
            best_loss = mylosses[epoch+1]

        counter += 1
        if counter%10 == 0:
            d_data.append(xD.numpy())
            a_data.append(xA.numpy())

        if np.abs(loss.numpy()) < tol:
            break
        
        if errorA < tol:
            print('Stopped training because of xA_new-xA_old =', errorA)
            break

        if errorD < tol:
            print('Stopped training because of xD_new-xD_old =', errorA)
            break
        
    t1 = time.time()
    dt = t1-t0

    print("\nApproximate value of chiA:", xA_best.numpy(), 
          "\nApproximate value of chiD:", xD_best.numpy(),
          "\nLoss - min #bosons on donor:", best_loss,
          "\nOptimizer Iterations:", OPT.iterations.numpy(), 
          "\nTraining Time:", dt,
          "\n"+40*"-",
          "\nParameters:",
          "\nOmega_A:", OMEGA_A.numpy(),
          "| Omega_D:", OMEGA_D.numpy(),
          "| N:", MAX_N.numpy(),
          "| Total timesteps:", MAX_T.numpy(),
          "\n"+40*"-")

    plt.plot(mylosses)
    saveFig(fig_id="loss", destination=fig_path)

    return xA_best.numpy(), xD_best.numpy(), a_data, d_data, CHIA, CHID

if __name__=="__main__":
    # change path to one with pre calculated values of avg_N
    min_n_path = os.path.join(os.getcwd(), 'data/coupling-0.1/tmax-25/avg_N/min_n_combinations')
    test_array = np.loadtxt(min_n_path)
    xA_plot, xD_plot = test_array[:,0].reshape(100,100), test_array[:,1].reshape(100,100)
    avg_n = test_array[:,2].reshape(100,100)
    
    
    data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
    createDir(data_path, replace=True)
    
    xa, xd, a, d, a_init, d_init = train(fig_path=data_path)
    titl = f'N={4}, tmax={25}, Initial (χA, χD) = {a_init.numpy(), d_init.numpy()}, λ={0.1}, ωA={-3}, ωD={3}'    
    
    x = np.array(np.array(d))
    y = np.array(np.array(a))
    figure2, ax2 = plt.subplots(figsize=(12,12))
    # plot the predictions of the optimizer
    plot2 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
    ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
    ax2.scatter(a_init, d_init, color='green', edgecolors='black', s=94, label='Initial Value', zorder=3)
    ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure2.colorbar(plot2)
    ax2.legend(prop={'size': 15})
    ax2.set_title(titl, fontsize=20)
    saveFig(fig_id="contour", destination=data_path)
