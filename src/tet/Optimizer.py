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
from HamiltonianLoss import Loss
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
        
        #! Import the parameters of the problem
        if const is None: self.const = constants.loadConstants()
        else: self.const = const

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

        # self.xA = None
        # self.xD = None

        self.vars = [None for _ in range(len(self.const['chis']))]

        self.DTYPE = TensorflowParams['DTYPE']
        
    def __call__(self, *args):
        self.init_chis = list(args)

        if self.DataExist: pass
        else:
            createDir(self.data_path, replace_query=True)
            self._train()
            
    @tf.function
    def compute_loss(self, lossClass):
        #return lossClass(self.xA, self.xD, site=self.target_site)
        return lossClass(*self.vars, site=self.target_site)

    def get_grads(self, lossClass):
        with tf.GradientTape() as t:
            t.watch([self.vars[k] for k in TensorflowParams['train_sites']])
            loss = self.compute_loss(lossClass)
        grads = t.gradient(loss, self.vars)
        del t
        return grads, loss

    @tf.function
    def apply_grads(self, lossClass):
        grads, loss = self.get_grads(lossClass)
        self.opt.apply_gradients(zip(grads, self.vars))
        return loss

    #! Running the optimizer given initial guesses for the trainable parameters
    def train(self, initial_chis):
        # Reset Optimizer
        K.clear_session()
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))
        K.set_value(self.opt.learning_rate, self.lr)

        # Define the object of the Loss class according to the current chiA,chiD.
        loss_ms = Loss(const=self.const)
        # Store the values of the loss function while proceeding 
        mylosses = []
        # Define the tolerance of the optimizer
        self.tol = TensorflowParams['tol']

        for i in range(len(self.const['chis'])):
            if self.vars[i] is None:
                self.vars[i] = tf.constant(value=initial_chis[i], dtype=self.DTYPE, name=f'chi{i}')
        # Non linearity parameters that produce the lowest loss function
        best_vars = [tf.constant(value=0, dtype=self.DTYPE) for _ in range(len(self.vars))]
        mylosses.append(self.max_n)
        best_loss = self.max_n
        counter = 0
        var_data = [[] for _ in range(len(self.vars))]
        var_error_count = [0 for _ in range(len(self.vars))]

        t0 = time.time()
        for epoch in range(self.iter):
            _vars = [self.vars[i].numpy() for i in range(len(self.vars))]
            loss = self.apply_grads(lossClass=loss_ms)
            if self.Print:
                if epoch%100 ==0: print(f'Loss:{loss.numpy()}, ',*[f'x{j}: {self.vars[j].numpy()}, ' for j in range(len(self.vars))], f', epoch:{epoch}')
            
            if loss.numpy()<=0.5:
                K.set_value(self.opt.learning_rate, 0.001)
                for i in range(len(self.vars)):
                    self.vars[i].assign(_vars[i])
            
            var_error = [np.abs(self.vars[i].numpy() - _vars[i]) for i in range(len(self.vars))]

            mylosses.append(loss.numpy())
            if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
                for i in range(len(self.vars)):
                    best_vars[i].assign(self.vars[i].numpy())
                best_loss = mylosses[epoch+1]

            counter += 1
            if counter%10 == 0:
                for k in range(len(self.vars)):
                    var_data[k].append(self.vars[k].numpy())

            if np.abs(loss.numpy()) < 0.1:
                break
            
            for j in range(len(self.vars)):
                if var_error[j] < self.tol:
                    var_error_count[j] += 1
                    if var_error_count[j] > 2:
                        if self.Print:
                            print(f'Stopped training because of x{j}_new-x{j}_old =', var_error[j])
                        break
            
        t1 = time.time()
        dt = t1-t0
        
        if self.Print:
            print
            (
                *[f"\nApproximate value of chiA: {best_vars[j]}" for j in range(len(self.vars))],
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
                "\n"+60*"-"
            )
        
        return mylosses, var_data, best_vars
    
    #! Run the optimizer for given initial guesses and save trajectories.
    def _train(self):
        mylosses, var_data, best_vars = self.train(self.init_chis)
        writeData(data=mylosses[1:], destination=self.data_path, name_of_file='losses.txt')
        for i in range(len(var_data)):
            writeData(data=var_data[i], destination=self.data_path, name_of_file=f'x{i}trajectory.txt')
            writeData(data=var_data[i], destination=self.data_path, name_of_file=f'x{i}trajectory.txt')

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

if __name__=="__main__":
    CONST = constants.constants
    opt = Optimizer(target_site=constants.solver_params['target'],
                    DataExist=False,
                    const=CONST)
    opt(*CONST['chis'])