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

from data_process import createDir, writeData, read_1D_data
from loss import Loss
import constants as constants
from constants import TensorflowParams

class Optimizer:
    def __init__(self,
                 target_site, 
                 DataExist,
                 const=None,
                 Plot=False, 
                 Print=True,
                 iterations=TensorflowParams['iterations'],
                 lr=TensorflowParams['lr'], 
                 data_path=os.path.join(os.getcwd(), 'data_optimizer')):
        
        if const is None: self.const = constants.loadConstants()
        else: self.const = const

        self.Npoints = self.const['Npoints']
        self.coupling = self.const['coupling']
        self.max_t = self.const['max_t']
        self.max_n = self.const['max_N']
        self.omegas = self.const['omegas']
        self.sites = self.const['sites']

        self.target_site = target_site
        self.DataExist = DataExist
        self.data_path = data_path
        self.plot = Plot
        self.iter = iterations
        self.lr = lr
        self.opt = tf.keras.optimizers.Adam()
        self.Print = Print

        self.xA = None
        self.xD = None

        self.DTYPE = TensorflowParams['DTYPE']
        
    def __call__(self, ChiAInitial, ChiDInitial):
        self.ChiAInitial = ChiAInitial
        self.ChiDInitial = ChiDInitial
        # self.CombinationPath = os.path.join(self.data_path, f'combination_{case}')

        if self.DataExist: pass
        else:
            createDir(self.data_path, replace_query=True)
            self._train()
            
    @tf.function
    def compute_loss(self, lossClass):
        return lossClass(self.xA, self.xD, site=self.target_site)

    def get_grads(self, lossClass):
        with tf.GradientTape() as t:
                t.watch([self.xA, self.xD])
                loss = self.compute_loss(lossClass)
        grads = t.gradient(loss, [self.xA, self.xD])
        del t
        return grads, loss

    @tf.function
    def apply_grads(self, lossClass):
        grads, loss = self.get_grads(lossClass)
        self.opt.apply_gradients(zip(grads, [self.xA, self.xD]))
        return loss

    def train(self, ChiAInitial, ChiDInitial):
        # Reset Optimizer
        K.clear_session()
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))
        K.set_value(self.opt.learning_rate, self.lr)
    
        loss_ms = Loss(const=self.const)
        
        mylosses = []
        tol = 1e-8
    
        if self.xA is None:
            self.xA = tf.Variable(initial_value=ChiAInitial, trainable=True, dtype=self.DTYPE, name='xA')
        if self.xD is None:
            self.xD = tf.Variable(initial_value=ChiDInitial, trainable=True, dtype=self.DTYPE, name='xD')
    
        xA_best = tf.Variable(initial_value=0, dtype=self.DTYPE, trainable=False)
        xD_best = tf.Variable(initial_value=0, dtype=self.DTYPE, trainable=False)
        mylosses.append(self.max_n)
        best_loss = self.max_n
        counter = 0
        d_data = []
        a_data = []
        a_error_count = 0
        d_error_count = 0

        t0 = time.time()
        for epoch in range(self.iter):
            xA_init = self.xA.numpy()
            xD_init = self.xD.numpy()
            loss = self.apply_grads(loss_ms)
            if self.Print:
                if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{self.xA.numpy()}, xD:{self.xD.numpy()}, epoch:{epoch}')
            
            if loss.numpy()<=0.5:
                K.set_value(self.opt.learning_rate, 0.001)
                self.xA.assign(value=xA_init)
                self.xD.assign(value=xD_init)
            
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
                    if self.Print:
                        print('Stopped training because of xA_new-xA_old =', errorA)
                    break

            if errorD < tol:
                d_error_count += 1
                if d_error_count > 2:
                    if self.Print:
                        print('Stopped training because of xD_new-xD_old =', errorA)
                    break
            
        t1 = time.time()
        dt = t1-t0
        
        if self.Print:
            print("\nApproximate value of chiA:", xA_best.numpy(), 
                "\nApproximate value of chiD:", xD_best.numpy(),
                "\nLoss - min #bosons on donor:", best_loss,
                "\nOptimizer Iterations:", self.opt.iterations.numpy(), 
                "\nTraining Time:", dt,
                "\n"+60*"-",
                "\nParameters:",
                "\nOmega_A:", self.omegas[-1],
                "| Omega_D:", self.omegas[0],
                "| N:", self.max_n,
                "| Sites: ", self.sites,
                "| Total timesteps:", self.max_t,
                "| Coupling Lambda:",self.coupling,
                "\n"+60*"-")
        
        return mylosses, a_data, d_data, xA_best.numpy(), xD_best.numpy()
    
    def _train(self):
        mylosses, a_data, d_data, xA_best, xD_best = self.train(self.ChiAInitial, self.ChiDInitial)
        writeData(data=mylosses[1:], destination=self.data_path, name_of_file='losses.txt')
        writeData(data=a_data, destination=self.data_path, name_of_file='xAtrajectory.txt')
        writeData(data=d_data, destination=self.data_path, name_of_file='xDtrajectory.txt')
        self.const['xA'] = str(xA_best)
        self.const['xD'] = str(xD_best)
        constants.dumpConstants(dict=self.const)

# ----------------------------- Multiprocess Helper Function ----------------------------- #

def mp_opt(i, ChiAInitial, ChiDInitial, iteration_path, const, target_site, lr, iterations):
    const = constants.loadConstants()
    data_path = os.path.join(os.getcwd(), f'{iteration_path}/data_optimizer_{i}')
    opt = Optimizer(target_site=target_site,
                    DataExist=False,
                    Print=False,
                    data_path=data_path,
                    const=const,
                    lr=lr,
                    iterations=iterations,
                    Plot=True)
    opt(ChiAInitial, ChiDInitial)
    
    # Load Data
    loss_data = read_1D_data(destination=data_path, name_of_file='losses.txt')
    a = const['xA']
    d = const['xD']
    print(f'Job {i}: Done')
    return np.array([a, d, np.min(loss_data)])
