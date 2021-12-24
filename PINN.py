#%%
import sys
assert sys.version_info >= (3, 5)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import multiprocessing as mp
import pandas as pd
from functools import partial

import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="jpg", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def write_data(data, destination, name_of_file):
    df = pd.DataFrame(data = data)
    _destination = os.path.join(destination, name_of_file)
    np.savetxt(_destination, df)

def read_2D_data(name_of_file):
    X,Y = [],[]
    for line in open(name_of_file, 'r'):
        lines = [i for i in line.split()]
        X.append(float(lines[0]))
        Y.append(float(lines[1]))
    return X,Y

def read_1D_data(name_of_file):
    data = []
    for line in open(name_of_file, 'r'):
        lines = [i for i in line.split()]
        data.append(float(lines[0]))
    return data

def construct_Hamiltonians(chiA,chiD):
  H,H_uncoupling = np.zeros((N + 1, N + 1), dtype=float),np.zeros((N + 1, N + 1), dtype=float)
  
  # Step 1a) Complete problem
  # i bosons at the donor
  for i in range(N + 1):
    for j in range(N + 1):
      # First term from interaction
      if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (N - i))
      # Second term from interaction
      if i == j + 1:H[i][j] = -coupling_lambda * np.sqrt(i * (N - i + 1))
        # Term coming from the two independent Hamiltonians
      if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (N - i) + 0.5 * chiA * (N - i) ** 2

  # Step 1b) Uncoupling problem
  np.fill_diagonal(H_uncoupling,np.diagonal(H))

  return H,H_uncoupling

#%%
class Opt_PertTheory(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # characteristic parameters of the problem
    self.sites = {"D":{"omegaD": -3, "chiD": 4}, "A":{"omegaA": 3, "chiA": -2}, "coupling_lambda": 0.001, "N":4}
    self.mylosses = []

  def An(self, i, xA, xD):
    f1 = self.sites["A"]["omegaA"] + 0.5 * xA * (2 * self.sites["N"] - 2 * i - 1) - self.sites["D"]["omegaD"] - 0.5 * xD * (2 * i + 1)
    return -tf.divide(tf.sqrt(float((i + 1) * (self.sites["N"] - i))),f1)

  def Bn(self, i, xA, xD):
    f2 = -self.sites["A"]["omegaA"] - 0.5 * xA * (2 * self.sites["N"] - 2 * i + 1) + self.sites["D"]["omegaD"] + 0.5 * xD * (2 * i - 1)
    return -tf.divide(tf.sqrt(float(i * (self.sites["N"] - i + 1))),f2)

  def Fn(self, i, xA, xD):
    return tf.divide(1,tf.sqrt(1+self.sites["coupling_lambda"]**2*self.An(i, xA, xD)**2 + self.Bn(i, xA, xD)**2))

  def Dhelp(self, i, xA, xD):
    f4 = 2*(self.sites["D"]["omegaD"] - self.sites["A"]["omegaA"] - xA * (self.sites["N"] - i + 1) + xD * (i - 1))
    return tf.divide(-self.Bn(i, xA, xD)*tf.sqrt(float((i-1)*(self.sites["N"]-i+2))),f4)

  def loss(self, xA, xD):
    N = self.sites["N"]
    # n == N
    PD = self.Fn(N, xA, xD)**4*(1+self.sites["coupling_lambda"]**2*self.Bn(N, xA, xD)**2+self.sites["coupling_lambda"]**4*self.Dhelp(N, xA, xD)**2)
    return PD
    
  def pred_plot(self, xA, xD):
    middleIndexA = (len(xA) - 1)//2
    middleIndexD = (len(xD) - 1)//2
    xA_min,xD_min = xA[middleIndexA]-2, xD[middleIndexD]-2
    xA_max,xD_max = xA[middleIndexA]+2, xD[middleIndexD]+2

    xA_grid = np.linspace(xA_min, xA_max, 1000)
    xD_grid = np.linspace(xD_min, xD_max, 1000)

    xA_plot, xD_plot = np.meshgrid(xA_grid, xD_grid)

    probs = self.loss(xA_plot, xD_plot)
    # create a surface plot with the rainbow color scheme
    figure, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
    plot = ax.plot_surface(xA_plot, xD_plot, probs, cmap='rainbow')
    ax.set_xlabel("xA", fontsize=20)
    ax.set_ylabel("xD", fontsize=20)
    ax.set_zlabel("$P_{D}$", fontsize=20)
    figure.colorbar(plot, shrink=0.5)
    # show the plot
    save_fig("pertubation_theory_loss")
    plt.show()

    x = np.array(xA)
    y = np.array(xD)
    figure2, ax2 = plt.subplots(figsize=(12,12))
    # plot the predictions of the optimizer
    plot2 = ax2.contourf(xA_plot, xD_plot, probs, levels=50, cmap='rainbow')
    ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
    ax2.scatter(self.sites["A"]["chiA"], self.sites["D"]["chiD"], color='green', edgecolors='black', s=94, label='Initial Value')
    ax2.set_xlabel("xA", fontsize=20)
    ax2.set_ylabel("xD", fontsize=20)
    figure2.colorbar(plot2)
    ax2.legend(prop={'size': 15})
    save_fig("pertubation_theory_loss_contour")
    plt.show()

  def train(self):
    tol = 1E-8
    max_iter = 1000
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    xA = tf.Variable(initial_value=self.sites["A"]["chiA"], trainable=True, dtype=tf.float32)
    xD = tf.Variable(initial_value=self.sites["D"]["chiD"], trainable=True, dtype=tf.float32)
    xA_best = tf.Variable(initial_value=0, dtype=tf.float32)
    xD_best = tf.Variable(initial_value=0, dtype=tf.float32)
    self.mylosses.append(1)
    best_loss = 1
    counter = 0
    d_data = []
    a_data = []

    t0 = time.time()
    for epoch in range(max_iter):
      xA_init = xA.numpy()
      xD_init = xD.numpy()
      
      with tf.GradientTape() as t:
        current_loss = self.loss(xA, xD)
      _train = opt.minimize(current_loss, var_list=(xA, xD), tape=t)
      # grads = t.gradient(current_loss, [xA, xD])
      # opt.apply_gradients(zip(grads, [xA, xD]))

      errorA = np.abs(xA.numpy() - xA_init)
      errorD = np.abs(xD.numpy() - xD_init)

      if errorA < tol:
        del t
        break

      if errorD < tol:
        del t
        break

      self.mylosses.append(abs(current_loss.numpy()))
      if self.mylosses[epoch+1] < min(list(self.mylosses[:epoch+1])):
        xA_best.assign(xA.numpy())
        xD_best.assign(xD.numpy())
        best_loss = self.mylosses[epoch+1]

      counter += 1
      if counter%10 == 0:
        d_data.append(xD.numpy())
        a_data.append(xA.numpy())

      if np.abs(current_loss.numpy()) < tol:
        del t
        break
        
      # if opt.iterations.numpy() > max_iter:
      #   del t
      #   break
    t1 = time.time()
    dt = t1-t0

    print("\nApproximate value of chiA:", xA_best.numpy(), 
          "\nApproximate value of chiD:", xD_best.numpy(),
          "\nLoss:", best_loss,
          "\nOptimizer Iterations:", opt.iterations.numpy(), 
          "\nTraining Time:", dt,
          "\n"+40*"-")

    self.pred_plot(a_data, d_data)
    
    # return xA_best.numpy(), xD_best.numpy(), best_loss, a_data, d_data
    return xA_best.numpy(), xD_best.numpy()

# %%
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# import gym

class RL_Test(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # self.
    # self.dense1 = Dense(5, activation="elu", input_shape=[n_inputs])

  def loss(self, xA, xD):
    return tf.math.tanh(xA**2 + xD**2)

  def plot(self):
    xa = tf.linspace(-2, 2, 100)
    xd = tf.linspace(-2, 2, 100)

    XA, XD = tf.meshgrid(xa, xd)
    Z = self.loss(XA, XD)

    figure, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
    ax.plot_surface(XA, XD, Z, cmap='rainbow')
    plt.show()
# %%
def bosons_at_donor_analytical(max_t,eigvecs,eigvals,initial_state):
  coeff_c = np.zeros(N+1,dtype=float)
  for i in range(N+1): 
    coeff_c[i] = np.vdot(eigvecs[:,i], initial_state)

  j_vectors = np.identity(N+1)

  coeff_b = np.zeros(eigvecs.shape)
  for j in range(N+1):
    for i in range(N+1):
      coeff_b[j,i] = np.vdot(j_vectors[:, j], eigvecs[:, i])

  avg_N = np.zeros(max_t+1,dtype=float)
  t = 0

  while True and t <= max_t:
    sum_m = 0
    for m in range(N+1):
      sum_i = 0
      for i in range(N+1):
        sum_i += coeff_c[i]*coeff_b[m,i]*np.exp(-1j*eigvals[i]*t)

      sum_k = 0
      for k in range(N+1):
        sum_k += coeff_c[k].conj()*coeff_b[m,k].conj()*np.exp(1j*eigvals[k]*t)*sum_i
    
      sum_m += sum_k*m
    #print("\rt={}".format(t), end = "")
    avg_N[t] = sum_m
    t += 1

  return avg_N

def mp_bosons_at_donor_analytical(max_t,eigvecs,eigvals,initial_state):
  coeff_c = np.zeros(N+1,dtype=float)
  for i in range(N+1): 
    coeff_c[i] = np.vdot(eigvecs[:,i], initial_state)

  j_vectors = np.identity(N+1)

  coeff_b = np.zeros(eigvecs.shape)
  for j in range(N+1):
    for i in range(N+1):
      coeff_b[j,i] = np.vdot(j_vectors[:, j], eigvecs[:, i])

  avg_N = np.zeros(max_t+1,dtype=float)
  exp_time = range(0, max_t+1)

  # for t in exp_time:
  #   with mp.Pool(4) as pool:
      
  
  # print(zip(eigvals, time, coeff_c, coeff_b))

  for t in exp_time:
    sum_m = _mp_avg_N_calc_m(eigvals, t, coeff_c, coeff_b, N)
    #print("\rt={}".format(t), end = "")
    avg_N[t] = sum_m

  return avg_N

def _mp_avg_N_calc_m(eigvals, t, coeff_c, coeff_b, N):
  sum_j = 0
  for j in range(N+1):
    sum_i = sum(coeff_c[i]*coeff_b[j,i]*np.exp(-1j*eigvals[i]*t) for i in range(N+1))
    sum_k = sum(coeff_c[k].conj()*coeff_b[j,k].conj()*np.exp(1j*eigvals[k]*t)*sum_i for k in range(N+1))
    sum_j += sum_k*j
  return sum_j


if __name__ == "__main__":
  N = 12
  omegaA,omegaD = 3,-3
  chiA,chiD = -0.5,0.5
  coupling_lambda = 0.001

  cwd = os.getcwd()
  new_data = f"{cwd}/new_data"
  try:
    os.mkdir(new_data)
  except OSError as error:
    print(error)
    while True:
        query = input("Directory exists, replace it? [y/n]")
        fl_1 = query[0].lower() 
        if query == '' or not fl_1 in ['y','n']: 
            print('Please answer with yes or no')
        else: break
    if fl_1 == 'n': sys.exit(0)

  dir_name = f"coupling-{coupling_lambda}"
  data_dest = os.path.join(new_data, dir_name)
  print("Writing values in:", data_dest)
  try:
    os.mkdir(data_dest)
  except OSError as error:
    print(error)
    while True:
        query = input("Directory exists, replace it? [y/n]")
        fl_2 = query[0].lower() 
        if query == '' or not fl_2 in ['y','n']: 
            print('Please answer with yes or no')
        else: break
    if fl_2 == 'n': sys.exit(0)

  def mp_execute(chiA,chiD, data_dir):
    # ----------- Step 1: Construct the Hamiltonian matrices of the problem -----------
    H, H_uncoupling  = construct_Hamiltonians(chiA,chiD)

    # ----------- Step 2: Solve the eigenvalue problem -----------
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    #Diagonal matrix. Easy to compute eigenvalues
    # eigenvalues_uncoupling, eigenvectors_uncoupling = linalg.eigh(H_uncoupling)

    #----------- Step 3: Initial state ----------- 
    initial_state = np.zeros(N+1,dtype=float)
    for n in range(N+1): initial_state[n] = np.exp(-(N-n)**2)
    initial_state = initial_state / np.linalg.norm(initial_state)
    t_max = 2000
    t_span = np.linspace(0,t_max,t_max+1)
    avg_ND_analytical = mp_bosons_at_donor_analytical(max_t=t_max,eigvecs=eigenvectors,eigvals=eigenvalues,initial_state=initial_state)
    # print(f'\nExecution time ={round(time.time() - start_time,4)} seconds')
    title_file = f'ND_analytical-λ={coupling_lambda}-χA={chiA}-χD={chiD}.txt'
    write_data(avg_ND_analytical, destination=data_dir, name_of_file=title_file)

  t1 = time.time()
  mp_execute(chiA, chiD, data_dest)
  t2 = time.time()
  dt = t2-t1
  print(f"Code took:{dt}secs to run")

  df = pd.read_csv(os.path.join(data_dest, os.listdir(data_dest)[0]))
  df.plot()
  save_fig("average_number_of_bosons")
# %%
