from audioop import avg
from cProfile import label
from math import comb
from sre_constants import SUCCESS
import sys
from IPython.display import display,Latex
from scipy import rand
from Execute import Execute
from sympy import Chi, Lambda, per, rad
assert sys.version_info >= (3,6)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from tet.data_process import createDir,writeData,read_1D_data
from tet.saveFig import saveFig
from itertools import combinations, product

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K
from scipy.optimize import curve_fit
# Set the parameters
DTYPE = tf.float32
POINTSBACKGROUND = 250

LAMBDA = tf.constant(0.1, dtype=DTYPE)
OMEGA_A = tf.constant(3, dtype=DTYPE)
OMEGA_D = tf.constant(-3, dtype=DTYPE)
MAX_N = tf.constant(1, dtype=DTYPE)
MAX_T = tf.constant(25, dtype=tf.int32)
DIM = int(tf.constant(MAX_N+1).numpy())

def PlotResults():

    plt.figure(figsize=(14,4))

    plt.subplot(1,2,1)
    plt.scatter(xA_plot,xD_plot,c='r',label='Pairs for TET')
    plt.plot(xA_plot_,a*xA_plot_+b,c='b',label = r"$\chi_D$ = {:.3f}$\chi_A$ + {:.3f} $\approx$ $\chi_A$ +2($\omega_A-\omega_D)$".format(a,b))
    #plt.plot(xA_plot_,a*xA_plot_+b,c='b',label = r"$\chi_A$ = ({:.f} $\pm$ {:.3f}) $\cdot \chi_D$+({:.3f} $\pm$ {:.3f})".format(
    #    a,errorA,b,errorb))
    plt.xlabel(r"$\chi_A$")
    plt.ylabel(r"$\chi_D$")
    plt.legend()
    

    plt.subplot(1,2,2)
    ChiAManiadis = (OMEGA_D.numpy()-OMEGA_A.numpy())/MAX_N.numpy()
    ChiDManiadis = -ChiAManiadis
    dataManiadis = Execute(chiA=ChiAManiadis,chiD = ChiDManiadis,
                          coupling_lambda=LAMBDA.numpy(),
                          omegaA=OMEGA_A.numpy(),omegaD=OMEGA_D.numpy(),
                          max_N=int(MAX_N.numpy()),max_t=MAX_T.numpy(),
                          data_dir=os.getcwd(),return_data=True)()
    plt.plot(np.arange(int(MAX_T.numpy())+1),dataManiadis,label=r"$(\chi_A,\chi_D)=({:.3f},{:.3f})$".format(
        ChiAManiadis,ChiDManiadis))
    plt.xlabel('Time')
    plt.ylabel(r"$<N_D(t)>$")
    plt.legend()
    plt.suptitle(r"$\lambda={:.3f},\omega_A={},\omega_D={}, N=1$".format(LAMBDA.numpy(),OMEGA_A.numpy(),
    OMEGA_D.numpy()))
    plt.savefig('FitPairs.pdf')
    #plt.show()
    plt.close()


    #Plot random case
    random_indices = np.random.randint(len(avg_n),size = 6)
    DataRandom = []
    for index in random_indices:
        data = Execute(chiA=xA_plot[index],chiD=xD_plot[index],coupling_lambda=LAMBDA.numpy(),
            omegaA=OMEGA_A.numpy(),omegaD=OMEGA_D.numpy(),max_N=int(MAX_N.numpy()),max_t=MAX_T.numpy(),
            data_dir=os.getcwd(),return_data=True)()
        DataRandom.append(data)
    
    plt.figure(figsize=(11,8))
    for k in range(len(random_indices)):
        plt.subplot(3,2,k+1)
        plt.plot(np.arange(int(MAX_T.numpy())+1),DataRandom[k],c='b',label=r"$(\chi_A,\chi_D)=({:.3f},{:.3f})$".format(
        xA_plot[random_indices[k]],xD_plot[random_indices[k]]))
        
        plt.ylabel(r"$<N_D(t)>$")
    

        if k == len(random_indices)-2 or k == len(random_indices)-1:plt.xlabel('Time')
        plt.legend()
    plt.savefig('RandomCases.pdf')
    #plt.show()



def func(x, a, b):
    return a * x+b



OPT = tf.keras.optimizers.Adam(learning_rate=0.01)

#Read the data
min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(LAMBDA.numpy())+'/tmax-'+
    str(MAX_T.numpy())+'/avg_N/min_n_combinations')
#min_n_path = os.path.join(os.getcwd(), 'data/coupling-0.1/tmax-25/avg_N/min_n_combinations')
test_array = np.loadtxt(min_n_path)
xA= test_array[:,0].reshape(-1,1)
xD= test_array[:,1].reshape(-1,1)
avg_n =test_array[:,2].reshape(-1,1)

# Find the TET cases
succeed_indices = np.where(avg_n<0.001)[0]
avg_n_ = avg_n[succeed_indices]
xA_plot_ = xA[succeed_indices]
xD_plot_ = xD[succeed_indices]
#Fix small format issues
xA_plot,xD_plot,avg_n = np.zeros(len(xA_plot_)),np.zeros(len(xA_plot_)),np.zeros(len(xA_plot_))
for i in range(len(xA_plot_)):
    xA_plot[i]= xA_plot_[i][0]
    xD_plot[i]= xD_plot_[i][0]
    avg_n[i] = avg_n_[i][0]


#Fit the data
popt, pcov = curve_fit(func,xA_plot, xD_plot)
error = np.sqrt(np.diag(pcov))
a,b = popt
errorA,errorb = error

#Plot results
PlotResults()
