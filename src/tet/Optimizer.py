import sys
import os
assert sys.version_info >= (3,6)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K

from tet.constants import Constants
from tet.data_process import createDir, writeData, read_1D_data
from tet.train import train
from tet.saveFig import saveFig

DTYPE = tf.float32
CONST = Constants()
LAMBDA = tf.constant(CONST.coupling, dtype=DTYPE)
OMEGA_A = tf.constant(CONST.omegaA, dtype=DTYPE)
OMEGA_D = tf.constant(CONST.omegaD, dtype=DTYPE)
MAX_N = tf.constant(CONST.max_N, dtype=DTYPE)
MAX_T = tf.constant(CONST.max_t, dtype=tf.int32)
POINTSBACKGROUND = CONST.plot_res

class Optimizer:
    def __init__(self, ChiAInitial, ChiDInitial, DataExist, Case, Plot=False):
        self.DataExist = DataExist
        self.ChiAInitial = ChiAInitial
        self.ChiDInitial = ChiDInitial
        self.data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
        self.CombinationPath = os.path.join(self.data_path,f'combination_{Case}')
        self.plot = Plot
        
    def __call__(self):
        if self.DataExist and self.plot: self.PlotResults()
        # If data exists according tot the user, dont do anything
        elif self.DataExist and not self.plot: pass
        
        else:
            createDir(self.data_path, replace=False)
            createDir(destination=self.CombinationPath,replace=True)
            self._train()
            if self.plot: self.PlotResults()
    
    def _train(self):
        mylosses, a_data, d_data, xA_best, xD_best = train(self.ChiAInitial, self.ChiDInitial)
        writeData(data=mylosses[1:],destination=self.CombinationPath,name_of_file='losses.txt')
        writeData(data=a_data,destination=self.CombinationPath,name_of_file='xAs Trajectory.txt')
        writeData(data=d_data,destination=self.CombinationPath,name_of_file='xDs Trajectory.txt')
        
    def PlotResults(self):
        # Load Background
        min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(LAMBDA.numpy())+'/tmax-'+
        str(MAX_T.numpy())+'/avg_N/min_n_combinations')
        test_array = np.loadtxt(min_n_path)
        xA_plot = test_array[:,0].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
        xD_plot = test_array[:,1].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
        avg_n = test_array[:,2].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
        
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
        titl = f'N={MAX_N.numpy()}, tmax={MAX_T.numpy()}, Initial (χA, χD) = {a_init, d_init}, λ={LAMBDA.numpy()}, ωA={OMEGA_A.numpy()}, ωD={OMEGA_D.numpy()}'    
        
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

# EXAMPLE CALL    
if __name__=="__main__":
    for i in range(2):
        opt = Optimizer(i, 2, DataExist=False, Plot=False, Case=i)
        opt()