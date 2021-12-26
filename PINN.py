import sys
from typing import Type
assert sys.version_info >= (3, 5)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import multiprocessing as mp
import pandas as pd
from functools import partial
import warnings

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

def construct_Hamiltonians(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N):
  H = np.zeros((max_N + 1, max_N + 1), dtype=float)
  
  # i bosons at the donor
  for i in range(max_N + 1):
    for j in range(max_N + 1):
      # First term from interaction
      if i == j - 1: H[i][j] = -coupling_lambda * np.sqrt((i + 1) * (max_N - i))
      # Second term from interaction
      if i == j + 1: H[i][j] = -coupling_lambda * np.sqrt(i * (max_N - i + 1))
        # Term coming from the two independent Hamiltonians
      if i == j: H[i][j] = omegaD * i + 0.5 * chiD * i ** 2 + omegaA * (max_N - i) + 0.5 * chiA * (max_N - i) ** 2

  return H

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

def mp_bosons_at_donor_analytical(max_t, max_N, eigvecs, eigvals, initial_state):
  '''
  Function that calculates the average number of bosons in the dimer system.
  The calculation is done based on the equation (25) in our report.

  INPUTS:
    max_t == int(), The maximum time of the experiment.
    max_N == int(), The number of bosons on the donor site.
    eigvecs == np.array(),
               shape == (max_N+1, max_N+1),
               The eigenvectors of the hamiltonian of the system as columns in a numpy array.
    eigvals == np.array(),
               shape == max_N+1,
               The eigenvalues of the hamiltonian of the system.
    initial_state == np.array(),
                     shape == max_N+1,
                     An initial state for the system defined as the normalised version of:
                     np.exp(-(max_N-n)**2) where n is the n-th boson at the donor site.

  OUTPUTS:
    avg_N == list(),
             len() == max_t
             The average number of bosons at the donor.
  '''
  coeff_c = np.zeros(max_N+1, dtype=float)
  for i in range(max_N+1): 
    coeff_c[i] = np.vdot(eigvecs[:,i], initial_state)

  coeff_b = eigvecs

  avg_N = np.zeros(max_t+1, dtype=float)
  _time = range(0, max_t+1)

  while True:
    query = input("Run with multiprocessing? [y/n] ")
    fl_1 = query[0].lower() 
    if query == '' or not fl_1 in ['y','n']: 
        print('Please answer with yes or no')
    else: break

  if fl_1 == 'y': 
    p = mp.Pool()
    _mp_avg_N_calc_partial = partial(_mp_avg_N_calc, max_N=max_N, eigvals=eigvals, coeff_c=coeff_c, coeff_b=coeff_b)
    avg_N = p.map(_mp_avg_N_calc_partial, _time)

  if fl_1 == 'n':
    for t in _time:
      avg_N[t] = _mp_avg_N_calc(t, max_N, eigvals, coeff_c, coeff_b)

  return avg_N

def _mp_avg_N_calc(t, max_N, eigvals, coeff_c, coeff_b):
  sum_j = 0
  for j in range(max_N+1):
    sum_i = sum(coeff_c*coeff_b[j,:]*np.exp(-1j*eigvals*t))
    sum_k = sum(coeff_c.conj()*coeff_b[j,:].conj()*np.exp(1j*eigvals*t)*sum_i)
    sum_j += sum_k*j
  return sum_j

def mp_execute(chiA,chiD, data_dir, max_N):
  problemHamiltonian = construct_Hamiltonians(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N)
  eigenvalues, eigenvectors = np.linalg.eigh(problemHamiltonian)

  initial_state = np.zeros(max_N+1,dtype=float)
  for n in range(max_N+1): initial_state[n] = np.exp(-(max_N-n)**2)
  initial_state = initial_state / np.linalg.norm(initial_state)

  t_max = 2000

  avg_ND_analytical = mp_bosons_at_donor_analytical(max_t=t_max, 
                                                    max_N=max_N, 
                                                    eigvecs=eigenvectors,
                                                    eigvals=eigenvalues,
                                                    initial_state=initial_state)

  title_file = f'ND_analytical-λ={coupling_lambda}-χA={chiA}-χD={chiD}.txt'
  write_data(avg_ND_analytical, destination=data_dir, name_of_file=title_file)

if __name__ == "__main__":
  warnings.filterwarnings('ignore', category=np.ComplexWarning)

  max_N = 12
  omegaA, omegaD = 3, -3
  chiA, chiD = -0.5, 0.5
  coupling_lambda = 0.001

  cwd = os.getcwd()
  new_data = f"{cwd}/new_data"
  try:
    os.mkdir(new_data)
  except OSError as error:
    print(error)
    while True:
        query = input("Directory exists, replace it? [y/n] ")
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
        query = input("Directory exists, replace it? [y/n] ")
        fl_2 = query[0].lower() 
        if query == '' or not fl_2 in ['y','n']: 
            print('Please answer with yes or no')
        else: break
    if fl_2 == 'n': sys.exit(0)

  t1 = time.time()
  mp_execute(chiA, chiD, data_dest, max_N=max_N)
  t2 = time.time()
  dt = t2-t1
  print(f"Code took: {dt:.3f}secs to run")

  df = pd.read_csv(os.path.join(data_dest, os.listdir(data_dest)[0]))
  df.plot()
  save_fig("average_number_of_bosons")