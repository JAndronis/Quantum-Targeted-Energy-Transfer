import sys
import os
import time
assert sys.version_info >= (3,6)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K

from tet.data_process import createDir, writeData, read_1D_data
from tet.saveFig import saveFig
from loss import Loss
import constants

DTYPE = tf.float32

class Optimizer:
    def __init__(self, 
                 ChiAInitial, 
                 ChiDInitial, 
                 DataExist, 
                 Case,
                 const=None,
                 Plot=False, 
                 iterations=200,
                 lr=0.1, 
                 data_path=os.path.join(os.getcwd(), 'data_optimizer_avgn')):
        
        if const is None:
            self.const = constants.loadConstants()
        else: self.const = const
        self.res = self.const['resolution']
        self.coupling = self.const['coupling']
        self.max_t = self.const['max_t']
        self.max_n = self.const['max_N']
        self.omegaA = self.const['omegaA']
        self.omegaD = self.const['omegaD']
        self.DataExist = DataExist
        self.ChiAInitial = ChiAInitial
        self.ChiDInitial = ChiDInitial
        self.data_path = data_path
        self.CombinationPath = os.path.join(self.data_path, f'combination_{Case}')
        self.plot = Plot
        self.iter = iterations
        self.lr = lr
        self.opt = tf.keras.optimizers.Adam()
        
    def __call__(self):
        if self.DataExist and self.plot: self.PlotResults()
        # If data exists according to the user, dont do anything
        elif self.DataExist and not self.plot: pass
        
        else:
            createDir(self.data_path, replace=False)
            createDir(destination=self.CombinationPath,replace=True)
            self._train()
            if self.plot: self.PlotResults()
            
    @tf.function
    def compute_loss(self, const):
        return Loss(const=const).loss(self.xA, self.xD)

    def get_grads(self, const):
        with tf.GradientTape() as t:
                t.watch([self.xA, self.xD])
                loss = self.compute_loss(const)
        grads = t.gradient(loss, [self.xA, self.xD])
        del t
        return grads, loss

    @tf.function
    def apply_grads(self, const):
        grads, loss = self.get_grads(const)
        self.opt.apply_gradients(zip(grads, [self.xA, self.xD]))
        return loss

    def train(self, ChiAInitial, ChiDInitial, const, max_iter=200, lr=0.01):
        # Reset Optimizer
        K.clear_session()
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))
        K.set_value(self.opt.learning_rate, lr)
        
        LAMBDA = const['coupling']
        OMEGA_A = const['omegaA']
        OMEGA_D = const['omegaD']
        MAX_N = const['max_N']
        MAX_T = const['max_t']
        
        mylosses = []
        tol = 1e-8
        self.xA = tf.Variable(initial_value=ChiAInitial, trainable=True, dtype=DTYPE)
        self.xD = tf.Variable(initial_value=ChiDInitial, trainable=True, dtype=DTYPE)
        xA_best = tf.Variable(initial_value=0, dtype=DTYPE, trainable=False)
        xD_best = tf.Variable(initial_value=0, dtype=DTYPE, trainable=False)
        mylosses.append(MAX_N)
        best_loss = MAX_N
        counter = 0
        d_data = []
        a_data = []
        a_error_count = 0
        d_error_count = 0

        t0 = time.time()
        for epoch in range(max_iter):
            xA_init = self.xA.numpy()
            xD_init = self.xD.numpy()
            loss = self.apply_grads(const)
            if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{self.xA.numpy()}, xD:{self.xD.numpy()}, epoch:{epoch}')
            
            if loss.numpy()<=0.5:
                K.set_value(self.opt.learning_rate, 0.01)
            
            errorA = np.abs(self.xA.numpy() - xA_init)
            errorD = np.abs(self.xD.numpy() - xD_init)

            mylosses.append(loss.numpy())
            if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
                xA_best.assign(self.xA.numpy())
                xD_best.assign(self.xD.numpy())
                best_loss = mylosses[epoch+1]

            counter += 1
            if counter%10 == 0:
                d_data.append(self.xD.numpy())
                a_data.append(self.xA.numpy())

            if np.abs(loss.numpy()) < 0.1:
                break
            
            if errorA < tol:
                a_error_count += 1
                if a_error_count > 2:
                    print('Stopped training because of xA_new-xA_old =', errorA)
                    break

            if errorD < tol:
                d_error_count += 1
                if d_error_count > 2:
                    print('Stopped training because of xD_new-xD_old =', errorA)
                    break
            
        t1 = time.time()
        dt = t1-t0

        print("\nApproximate value of chiA:", xA_best.numpy(), 
            "\nApproximate value of chiD:", xD_best.numpy(),
            "\nLoss - min #bosons on donor:", best_loss,
            "\nOptimizer Iterations:", self.opt.iterations.numpy(), 
            "\nTraining Time:", dt,
            "\n"+40*"-",
            "\nParameters:",
            "\nOmega_A:", OMEGA_A,
            "| Omega_D:", OMEGA_D,
            "| N:", MAX_N,
            "| Total timesteps:", MAX_T,
            "| Coupling Lambda:",LAMBDA,
            "\n"+40*"-")
        
        return mylosses, a_data, d_data, xA_best.numpy(), xD_best.numpy()
    
    def _train(self):
        mylosses, a_data, d_data, xA_best, xD_best = self.train(self.ChiAInitial, self.ChiDInitial, \
            const=self.const, max_iter=self.iter, lr=self.lr)
        writeData(data=mylosses[1:], destination=self.CombinationPath, name_of_file='losses.txt')
        writeData(data=a_data, destination=self.CombinationPath, name_of_file='xAtrajectory.txt')
        writeData(data=d_data, destination=self.CombinationPath, name_of_file='xDtrajectory.txt')
        self.const['xA'] = str(xA_best)
        self.const['xD'] = str(xD_best)
        constants.dumpConstants(dict=self.const)
        
    def PlotResults(self):
        # Load Background
        min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(self.coupling)+'/tmax-'+
        str(self.max_n)+'/avg_N/min_n_combinations')
        test_array = np.loadtxt(min_n_path)
        xA_plot = test_array[:,0].reshape(self.res, self.res)
        xD_plot = test_array[:,1].reshape(self.res, self.res)
        avg_n = test_array[:,2].reshape(self.res, self.res)
        
        # Load Data
        loss_data = read_1D_data(destination=self.CombinationPath,name_of_file='losses.txt')
        a = read_1D_data(destination=self.CombinationPath,name_of_file='xAs Trajectory.txt')
        d = read_1D_data(destination=self.CombinationPath,name_of_file='xDs Trajectory.txt')
        a_init = self.ChiAInitial
        d_init = self.ChiDInitial
        
        # Plot Loss
        figure1, ax1 = plt.subplots()
        ax1.plot(loss_data[1:])
        saveFig(fig_id="loss", destination=self.CombinationPath)
        
        #Plot heatmaps with optimizer predictions
        titl = f'N={self.max_n}, tmax={self.max_t}, Initial (χA, χD) = {a_init, d_init},\
            λ={self.coupling}, ωA={self.omegaA}, ωD={self.omegaD}'    
        
        x = np.array(np.array(d))
        y = np.array(np.array(a))
        figure2, ax2 = plt.subplots(figsize=(12,12))
        plot2 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
        ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
        ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, label='Initial Value', zorder=3)
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot2)
        ax2.legend(prop={'size': 15})
        ax2.set_title(titl, fontsize=20)
        saveFig(fig_id="contour", destination=self.CombinationPath)
