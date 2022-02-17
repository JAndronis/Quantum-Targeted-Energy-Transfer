#%%
import sys
assert sys.version_info >= (3,6)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tet.data_process import createDir
from tet.saveFig import saveFig

import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras

class Opt_PertTheory(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# characteristic parameters of the problem
		self.sites = {"D":{"omegaD": -3, "chiD": 0.55}, "A":{"omegaA": 3, "chiA": 0.45}, "coupling_lambda": 0.1, "N":12}
		self.mylosses = []

	def An(self, i, xA, xD):
		f1 = self.sites["A"]["omegaA"] + 0.5 * xA * (2 * self.sites["N"] - 2 * i - 1) - self.sites["D"]["omegaD"] - 0.5 * xD * (2 * i + 1)
		return -tf.divide(tf.sqrt(float((i + 1) * (self.sites["N"] - i))),f1)

	def Bn(self, i, xA, xD):
		f2 = -self.sites["A"]["omegaA"] - 0.5 * xA * (2 * self.sites["N"] - 2 * i + 1) + self.sites["D"]["omegaD"] + 0.5 * xD * (2 * i - 1)
		return -tf.divide(tf.sqrt(float(i * (self.sites["N"] - i + 1))),f2)

	#Define Cn2 to avoid problems with negative input in sqrt later
	def Cn2(self, i, xA, xD):
		f3 = 2*(self.sites["A"]["omegaA"] - self.sites["D"]["omegaD"] + xA * (self.sites["N"] - i - 1) - xD * (i + 1))
		return tf.divide(self.An(i, xA, xD)**2*abs((i+2)*(self.sites["N"]-i-1) )**2 , f3**2)
	
	def Dn(self, i, xA, xD):
		f4 = 2*(self.sites["D"]["omegaD"] - self.sites["A"]["omegaA"] - xA * (self.sites["N"] - i + 1) + xD * (i - 1))
		return tf.divide(-self.Bn(i, xA, xD)*tf.sqrt(float((i-1)*(self.sites["N"]-i+2))),f4)

	def Fn(self, i, xA, xD):
		return tf.divide(1,tf.sqrt(1+self.sites["coupling_lambda"]**2*(tf.abs(self.An(i, xA, xD))**2 + tf.abs(self.Bn(i, xA, xD))**2)+self.sites["coupling_lambda"]**4*(tf.abs(self.Cn2(i, xA, xD))+tf.abs(self.Dn(i, xA, xD))**2)))

	def loss(self, xA, xD):
		N = self.sites["N"]
		# n == N
		PD = self.Fn(N, xA, xD)**4*(1+self.sites["coupling_lambda"]**2*self.Bn(N, xA, xD)**2+self.sites["coupling_lambda"]**4*self.Dn(N, xA, xD)**2)
		return PD
		
	def pred_plot(self, xA, xD):
		cwd = os.getcwd()
		data = f"{cwd}/data-optimizer"

		coupling_lambda = self.sites["coupling_lambda"]

		coupling_dir = f"coupling-{coupling_lambda}"
		coupling_dir_path = os.path.join(data, coupling_dir)

		createDir(data, replace=False)
		createDir(coupling_dir_path, replace=True)

		middleIndexA = (len(xA) - 1)//2
		middleIndexD = (len(xD) - 1)//2
		xA_min,xD_min = xA[middleIndexA]-4, xD[middleIndexD]-4
		xA_max,xD_max = xA[middleIndexA]+4, xD[middleIndexD]+4

		xA_grid = np.linspace(xA_min, xA_max, 1000)
		xD_grid = np.linspace(xD_min, xD_max, 1000)

		xA_plot, xD_plot = np.meshgrid(xA_grid, xD_grid)

		probs = self.loss(xA_plot, xD_plot)
		# create a surface plot with the rainbow color scheme
		figure, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
		plot = ax.plot_surface(xA_plot, xD_plot, probs, cmap='rainbow')
		ax.set_xlabel(r"$\chi_{A}$", fontsize=20)
		ax.set_ylabel(r"$\chi_{D}$", fontsize=20)
		ax.set_zlabel(r"$P_{D}$", fontsize=20)
		figure.colorbar(plot, shrink=0.5)
		# show the plot
		saveFig(fig_id="pertubation_theory_loss", destination=coupling_dir_path)
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
		ax2.set_xlabel(r"$\chi_{A}$", fontsize=20)
		ax2.set_ylabel(r"$\chi_{D}$", fontsize=20)
		figure2.colorbar(plot2)
		ax2.legend(prop={'size': 15})
		saveFig(fig_id="pertubation_theory_loss_contour", destination=coupling_dir_path)
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
# %%
