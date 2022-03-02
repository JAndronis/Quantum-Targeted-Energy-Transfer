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
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

# constants
DTYPE = tf.complex64

class Opt_PertTheory():
    def __init__(self):
        # characteristic parameters of the problem
        self.chiD = tf.constant(0, dtype=tf.float32)
        self.chiA = tf.constant(0, dtype=tf.float32)
        self.coupling_lambda = tf.constant(0.1, dtype=tf.float32)
        self.omegaA = tf.constant(-3, dtype=tf.float32)
        self.omegaD = tf.constant(3, dtype=tf.float32)
        self.max_N = tf.constant(4, dtype=tf.float32)
        self.dim = tf.constant(5, dtype=tf.int32)
        self.max_t = tf.constant(25, dtype=tf.float32)
        self.mylosses = []
        
        self.initial_state = tf.zeros(self.dim, dtype=DTYPE)
        initial_indices = []
        initial_updates = []
        for n in range(self.dim):
            initial_indices.append([n])
            i = tf.cast(n, dtype=tf.float32)
            initial_updates.append(tf.exp(-(self.max_N-i)**2))
        self.initial_state = tf.tensor_scatter_nd_update(self.initial_state, initial_indices, initial_updates)
        self.initial_state = self.initial_state / tf.linalg.norm(self.initial_state)

    def createHamiltonian(self, chiA, chiD):
        h = tf.zeros((self.dim, self.dim), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(self.dim):
            n = tf.cast(i, dtype=tf.float32)
            for j in range(self.dim):
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
        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)
        
        # dim = tf.cast(self.max_N, dtype=tf.int32)
        coeff_c = tf.zeros(self.dim, dtype=tf.complex64)
        c_indices = []
        c_updates = []
        for i in range(self.dim): 
            c_indices.append([i])
            c_updates.append(tf.tensordot(eigvecs[:,i], self.initial_state, 1))
        
        coeff_c = tf.tensor_scatter_nd_update(coeff_c, c_indices, c_updates)
        coeff_b = eigvecs
        self.coeff_c = coeff_c
        self.coeff_b = coeff_b
        self.eigvals = eigvals
    
    def computeAverage(self):
        avg_N = []
        _time = range(0, tf.cast(self.max_t, dtype=tf.int32).numpy()+1)
        last = tf.Variable(4.0, trainable=False)

        for t in _time:
            avg_N.append(self._computeAverageCalculation(t))
            # if tf.math.real(avg_N[-1].numpy()) > last: return tf.math.real(avg_N[-2])
            # last = tf.math.real(avg_N[-1].numpy())
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
        # avg_N = tf.math.log(avg_N)
        return avg_N
    
    def train(self):
        tol = 1e-8
        max_iter = 10
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        xA = tf.Variable(initial_value=self.chiA, trainable=True, dtype=tf.float32)
        xD = tf.Variable(initial_value=self.chiD, trainable=True, dtype=tf.float32)
        xA_best = tf.Variable(initial_value=0, dtype=tf.float32)
        xD_best = tf.Variable(initial_value=0, dtype=tf.float32)
        self.mylosses.append(4)
        best_loss = 4
        counter = 0
        d_data = []
        a_data = []

        # return self.loss(tf.constant(1.549), tf.constant(-1.49))

        t0 = time.time()
        for epoch in range(max_iter):
            xA_init = xA.numpy()
            xD_init = xD.numpy()

            with tf.GradientTape() as t:
                t.watch([xA, xD])
                current_loss = self.loss(xA, xD)
            # _train = opt.minimize(current_loss, var_list=(xA, xD), tape=t)
            grads = t.gradient(current_loss, [xA, xD])
            opt.apply_gradients(zip(grads, [xA, xD]))
            
            if epoch%100 ==0: print(f'Loss:{current_loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
            # print(f'Loss:{current_loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
            
            errorA = np.abs(xA.numpy() - xA_init)
            errorD = np.abs(xD.numpy() - xD_init)

            self.mylosses.append(abs(current_loss.numpy()))
            if self.mylosses[epoch+1] < min(list(self.mylosses[:epoch+1])):
                xA_best.assign(xA.numpy())
                xD_best.assign(xD.numpy())
                best_loss = self.mylosses[epoch+1]

            counter += 1
            if counter%5 == 0:
                d_data.append(xD.numpy())
                a_data.append(xA.numpy())

            if np.abs(current_loss.numpy()) < 0.1:
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

        d_data.append(xD.numpy())
        a_data.append(xA.numpy())
        
        print("\nApproximate value of chiA:", xA_best.numpy(), 
                    "\nApproximate value of chiD:", xD_best.numpy(),
                    "\nLoss:", best_loss,
                    "\nOptimizer Iterations:", opt.iterations.numpy(), 
                    "\nTraining Time:", dt,
                    "\n"+40*"-")

        plt.plot(self.mylosses)
        plt.title('Loss Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Predicted Number of bosons on donor')
        saveFig(fig_id="Loss Plot", destination=data_path)

        # return xA_best.numpy(), xD_best.numpy(), best_loss, a_data, d_data
        return xA_best.numpy(), xD_best.numpy(), a_data, d_data, self.chiA, self.chiD

if __name__=="__main__":
    # change path to one with pre calculated values of avg_N
    min_n_path = os.path.join(os.getcwd(), 'data/coupling-0.1/tmax-25/avg_N/min_n_combinations')
    test_array = np.loadtxt(min_n_path)
    xA_plot, xD_plot = test_array[:,0].reshape(100,100), test_array[:,1].reshape(100,100)
    avg_n = test_array[:,2].reshape(100,100)
    
    
    data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
    createDir(data_path, replace=True)
    
    xA_best, xD_best, a, d, a_init, d_init = Opt_PertTheory().train()
    titl = f'N={4}, tmax={25}, Initial (χA, χD) = {a_init.numpy(), d_init.numpy()}, λ={0.1}, ωA={-3}, ωD={3}'    
    
    x = np.array(np.array(a))
    y = np.array(np.array(d))
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
    saveFig(fig_id="Optimizer_AvgN_loss_contour", destination=data_path)
