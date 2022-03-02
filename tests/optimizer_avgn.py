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
CHIA = tf.constant(0, dtype=tf.float32)
CHID = tf.constant(0, dtype=tf.float32)
LAMBDA = tf.constant(1, dtype=tf.float32)
OMEGA_A = tf.constant(3, dtype=tf.float32)
OMEGA_D = tf.constant(-3, dtype=tf.float32)
MAX_N = tf.constant(4, dtype=tf.float32)
MAX_T = tf.constant(25, dtype=tf.float32)

OPT = tf.keras.optimizers.Adam(learning_rate=0.05)

class Loss:
    def __init__(self):
        self.coupling_lambda = LAMBDA
        self.omegaA = OMEGA_A
        self.omegaD = OMEGA_D
        self.max_N = MAX_N
        self.max_t = MAX_T
        
        dim = tf.cast(self.max_N, dtype=tf.int32)
        self.initial_state = tf.zeros(4+1,dtype=DTYPE)
        initial_indices = []
        initial_updates = []
        for n in range(4+1):
            initial_indices.append([n])
            i = tf.cast(n, dtype=tf.float32)
            initial_updates.append(tf.exp(-(self.max_N-i)**2))
        self.initial_state = tf.tensor_scatter_nd_update(self.initial_state, initial_indices, initial_updates)
        self.initial_state = self.initial_state / tf.linalg.norm(self.initial_state)

    def __call__(self, xA, xD):
        return self.loss(xA, xD)

    def createHamiltonian(self, chiA, chiD):
        h = tf.zeros((4+1, 4+1), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(4 + 1):
            n = tf.cast(i, dtype=tf.float32)
            for j in range(4 + 1):
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
        coeff_c = tf.zeros(4+1, dtype=tf.complex64)
        c_indices = []
        c_updates = []
        for i in range(4+1): 
            c_indices.append([i])
            c_updates.append(tf.tensordot(eigvecs[:,i], self.initial_state, 1))
        
        coeff_c = tf.tensor_scatter_nd_update(coeff_c, c_indices, c_updates)
        coeff_b = eigvecs
        return coeff_c, coeff_b, eigvals
    
    def computeAverage(self, c, b, e):
        _time = MAX_T
        n = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for t in tf.range(_time):
            t = tf.cast(t, dtype=tf.complex64)
            sum_j = tf.cast(0.0, dtype=tf.complex64)
            for j in tf.range(4+1):
                temp_b = tf.gather(b, [j], axis=1)
                temp_c = c
                temp_e = e
                sum_i = tf.reduce_sum(temp_c*temp_b*tf.exp(-tf.complex(0.0,1.0)*temp_e*t))
                sum_k = tf.reduce_sum(temp_c*temp_b*tf.exp(tf.complex(0.0,1.0)*temp_e*t))*sum_i
                j = tf.cast(j, dtype=tf.complex64)
                sum_j = tf.add(sum_j, sum_k*j)
            sum_j = tf.math.real(sum_j)
            # sum_j = tf.math.abs(sum_j)
            n = n.write(n.size(), tf.cast(sum_j, dtype=tf.float32))

        # n = tf.reduce_min(n)
        return n

    # def _computeAverageCalculation(self, t, c, b, e):
        
    def loss(self, xA, xD):
        coeff_c, coeff_b, vals = self.coeffs(xA, xD)
        avg_N_list = self.computeAverage(coeff_c, coeff_b, vals)
        avg_N = tf.reduce_min(avg_N_list.stack())
        return avg_N

@tf.function
def compute_loss(xA, xD):
    return Loss().loss(xA, xD)

def get_grads(xA, xD):
    with tf.GradientTape() as t:
            t.watch([xA, xD])
            loss = compute_loss(xA, xD)
    # _train = opt.minimize(loss, var_list=(xA, xD), tape=t)
    grads = t.gradient(loss, [xA, xD])
    del t
    return grads, loss
    
@tf.function
def apply_grads(xA, xD):
    grads, loss = get_grads(xA, xD)
    OPT.apply_gradients(zip(grads, [xA, xD]))
    return loss
    

def train():
    mylosses = []
    tol = 1e-8
    max_iter = 1000
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

    # return print(Loss().loss(xA, xD))

    t0 = time.time()
    for epoch in range(max_iter):
        xA_init = xA.numpy()
        xD_init = xD.numpy()
        loss = apply_grads(xA, xD)
        
        if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        # print(f'Loss:{loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        
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

    plt.plot(mylosses[1:])
    plt.show()

    # pred_plot(a_data, d_data)

    return xA_best.numpy(), xD_best.numpy(), a_data, d_data

def pred_plot(xA, xD):

    cwd = os.getcwd()
    data = f"{cwd}/data-optimizer"

    coupling_lambda = LAMBDA.numpy()

    coupling_dir = f"coupling-{coupling_lambda}"
    coupling_dir_path = os.path.join(data, coupling_dir)

    createDir(data, replace=False)
    createDir(coupling_dir_path, replace=True)

    middleIndexA = (len(xA) - 1)//2
    middleIndexD = (len(xD) - 1)//2
    xA_min,xD_min = xA[middleIndexA]-4, xD[middleIndexD]-4
    xA_max,xD_max = xA[middleIndexA]+4, xD[middleIndexD]+4

    xA_grid = np.linspace(xA_min, xA_max, 100, dtype=np.float32)
    xD_grid = np.linspace(xD_min, xD_max, 100, dtype=np.float32)

    xA_plot, xD_plot = np.meshgrid(xA_grid, xD_grid)

    probs = np.zeros(shape=(len(xA_plot), len(xD_plot)), dtype=np.float32)

    for i in tf.range(len((xA_plot))):
        for j in tf.range(len((xD_plot))):
            x1 = tf.cast(i, dtype=tf.float32)
            x2 = tf.cast(j, dtype=tf.float32)
            probs[i,j] = compute_loss(x1, x2).numpy()
    
    x = np.array(xD)
    y = np.array(xA)
    figure2, ax2 = plt.subplots(figsize=(12,12))
    # plot the predictions of the optimizer
    plot2 = ax2.contourf(xD_plot, xA_plot, probs, levels=50, cmap='rainbow')
    ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
    ax2.scatter(CHIA.numpy(), CHID.numpy(), color='green', edgecolors='black', s=94, label='Initial Value')
    ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure2.colorbar(plot2)
    ax2.legend(prop={'size': 15})
    saveFig(fig_id="pertubation_theory_loss_contour", destination=coupling_dir_path)
    plt.show()

if __name__=="__main__":
    xa, xd, a, d = train()
    # xA = np.linspace(-4,4,100)
    # xD = xA

    # print(compute_loss(tf.constant(-4.0), tf.constant(3.0)))
    
    # test_data = np.loadtxt('/Users/jasonandronis/Documents/GitHub/Thesis/tests/data/coupling-1/tmax-25/avg_N/min_n_combinations')
    # XA, XD = test_data[:,0].reshape(100,100), test_data[:,1].reshape(100,100)
    # avg_n = test_data[:,2].reshape(100,100)

    # cwd = os.getcwd()
    # data = f"{cwd}/data-optimizer"

    # coupling_dir = f"coupling-{LAMBDA.numpy()}"
    # coupling_dir_path = os.path.join(data, coupling_dir)

    # t_dir = f"tmax-{MAX_T.numpy()}"
    # t_dir_path = os.path.join(coupling_dir_path, t_dir)

    # data_dest = os.path.join(t_dir_path, "avg_N")

    # createDir(data, replace=False)
    # createDir(coupling_dir_path, replace=False)
    # createDir(t_dir_path)
    # createDir(data_dest)
    
    # titl = f'N={4}, tmax = {25}, # points (χA, χD) = {100, 100}, λ={LAMBDA}, ωA={OMEGA_A}, ωD={OMEGA_D}'
    # figure2, ax2 = plt.subplots(figsize=(12,12))
    # plot2 = ax2.contourf(avg_n,cmap = 'rainbow',extent=[min(xD),max(xD),min(-xA),max(-xA)], levels=50)
    # plt3 = ax2.scatter(a, d, c='black')
    # plt4 = ax2.scatter(xa, xd, c='blue')
    # ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
    # ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
    # figure2.colorbar(plot2)
    # ax2.set_title(titl, fontsize=20)
    # # saveFig(titl+' - contourplot', os.getcwd())
    # plt.show()
