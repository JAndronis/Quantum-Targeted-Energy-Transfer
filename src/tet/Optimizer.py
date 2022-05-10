import sys
import os
import time
assert sys.version_info >= (3,6)


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#tf.get_logger().setLevel('WARNING')
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

        self.vars = [None for _ in range(len(self.const['chis']))]

        self.DTYPE = TensorflowParams['DTYPE']
        
    def __call__(self, *args):
        self.init_chis = list(args)

        if self.DataExist: pass
        else:
            createDir(self.data_path, replace_query=True)
            self._train()
            
    @tf.function
    #! Compute the loss function
    def compute_loss(self, lossClass):
        return lossClass(*self.vars, site=self.target_site)

    #! Get the gradients
    def get_grads(self, lossClass):
        with tf.GradientTape() as t:
            #t watches the trainable parameters only by default
            loss = self.compute_loss(lossClass)
        grads = t.gradient(loss, self.vars)
        del t
        return grads, loss

    @tf.function
    #! Apply the gradients
    def apply_grads(self, lossClass):
        grads, loss = self.get_grads(lossClass)
        self.opt.apply_gradients(zip(grads, self.vars))
        return loss

    #! Running the optimizer given initial guesses for the trainable parameters
    def train(self, initial_chis):
        #* Reset Optimizer
        K.clear_session()
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))
        K.set_value(self.opt.learning_rate, self.lr)

        #* Define the object of the Loss class according to the current trainable parameters.
        #? ATTENTION
        loss_ms = Loss(const=self.const)

        # Save the values of the loss function while proceeding 
        mylosses = []

        # Define the tolerance of the optimizer
        self.tol = TensorflowParams['tol']

        # Discriminate trainable and non-trainable parameters
        for i in range(len(self.const['chis'])):
            if self.vars[i] is None:
                # Trainable ones
                if i in TensorflowParams['train_sites']:
                    self.vars[i] = tf.Variable(initial_value=initial_chis[i], dtype=self.DTYPE, name=f'chi{i}', trainable=True)
                #Non-trainable ones
                else:
                    self.vars[i] = tf.Variable(initial_value=initial_chis[i], dtype=self.DTYPE, name=f'chi{i}', trainable=False)
        
        # Non linearity parameters that produce the lowest loss function
        best_vars = [tf.Variable(initial_value=0, dtype=self.DTYPE, trainable=False) for _ in range(len(self.vars))]
        # Add the initial value of the loss function to ensure that following condition statements will apply
        mylosses.append(self.max_n)
        
        best_loss = self.max_n
        # Keep the changes for each parameters. Non-trainable parameters are not supposed to change
        counter = 0
        var_data = [[] for _ in range(len(self.vars))]
        # A help list for controlling the interuption of the iterarive procedure
        var_error_count = [0 for _ in range(len(self.vars))]

        t0 = time.time()
        for epoch in range(self.iter):
            # Help list with the variables before applying gradients
            _vars = [self.vars[i].numpy() for i in range(len(self.vars))]
            # Compute the loss function and obtain the updated variables
            loss = self.apply_grads(lossClass=loss_ms)

            # Set a repetition rate for displaying the progress 
            if self.Print:
                if epoch%100 ==0: print(f'Loss:{loss.numpy()}, ',*[f'x{j}: {self.vars[j].numpy()}, ' for j in range(len(self.vars))], f', epoch:{epoch}')
            
            # Reduce the learning rate when being close to TET
            if loss.numpy()<=0.5:
                K.set_value(self.opt.learning_rate, self.lr/100)
                for i in range(len(self.vars)):
                    self.vars[i].assign(_vars[i])
            
            #? If the above condition statement is true, wouldn't var_errror be 0?
            # Manual check for the progress of each variable
            var_error = [np.abs(self.vars[i].numpy() - _vars[i]) for i in range(len(self.vars))]
            # Save the new value of the loss function
            mylosses.append(loss.numpy())

            # Keep the minimum loss and the corresponding parameters
            if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
                for i in range(len(self.vars)):
                    best_vars[i].assign(self.vars[i].numpy())
                best_loss = mylosses[epoch+1]

            # Keep the changes of the variables per 10 steps
            counter += 1
            if counter%10 == 0:
                for k in range(len(self.vars)):
                    var_data[k].append(self.vars[k].numpy())

            # Interrupt in case of TET 
            if np.abs(loss.numpy()) < 0.1:
                break
            
            # Interrupt in case of non-progress
            for j in range(len(self.vars)):
                # Non trainable parameters change slightly
                if var_error[j] < self.tol and j in TensorflowParams['train_sites']:
                    var_error_count[j] += 1
                    if var_error_count[j] > 2:
                        if self.Print:
                            print(f'Stopped training because of x{j}_new-x{j}_old =', var_error[j])
                        t1 = time.time()
                        dt = t1-t0
                        if self.Print:
                            print(
                                *[f"\nApproximate value of chiA: {best_vars[j].numpy()}" for j in range(len(self.vars))],
                                "\nLoss - min #bosons on donor:", best_loss,
                                "\nOptimizer Iterations:", self.opt.iterations.numpy(), 
                                "\nTraining Time:", dt,
                                "\n"+60*"-",
                                "\nParameters:",
                                "\nOmegas", self.omegas,
                                "| N:", self.max_n,
                                "| Sites: ", self.sites,
                                "| Total timesteps:", self.max_t,
                                "| Coupling Lambda:",self.coupling,
                                "\n"+60*"-"
                            )
                        return mylosses, var_data, best_vars

        t1 = time.time()
        dt = t1-t0
        
        #Print the outcome of the optimizer
        if self.Print:
            print(
                *[f"\nApproximate value of chiA: {best_vars[j].numpy()}" for j in range(len(self.vars))],
                "\nLoss - min #bosons on donor:", best_loss,
                "\nOptimizer Iterations:", self.opt.iterations.numpy(), 
                "\nTraining Time:", dt,
                "\n"+60*"-",
                "\nParameters:",
                "\nOmegas", self.omegas,
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

        # Save the evolution of the values of the loss function
        writeData(data=mylosses[1:], destination=self.data_path, name_of_file='losses.txt')
        # Save the optimal parameters
        to_write_vars = self.const['chis']
        for (index,case) in zip(TensorflowParams['train_sites'],best_vars):to_write_vars[index] = case
        writeData(data=to_write_vars, destination=self.data_path, name_of_file='optimalvars.txt')
        for i in range(len(var_data)):
            # Save the trajectories in a file
            writeData(data=var_data[i], destination=self.data_path, name_of_file=f'x{i}trajectory.txt')
        self.const['chis'] = [f'{best_vars[i].numpy()}' for i in range(len(best_vars))]
        constants.dumpConstants(dict=self.const)

# ----------------------------- Multiprocess Helper Function ----------------------------- #

def mp_opt(i, combination, iteration_path, const, target_site, lr, iterations):
    #! Import the parameters of the problem
    const = constants.loadConstants()
    data_path = os.path.join(os.getcwd(), f'{iteration_path}/data_optimizer_{i}')

    #! Create the current optimizer
    opt = Optimizer(target_site=target_site,
                    DataExist=False,
                    Print=False,
                    data_path=data_path,
                    const=const,
                    lr=lr,
                    iterations=iterations,
                    Plot=True)

    #! Call the optimizer with chis including the given initial guesses
    input_chis = const['chis']
    # Update the list with the initial guesses of the optimizer
    for (index,case) in zip(TensorflowParams['train_sites'],combination):input_chis[index] = case
    opt(*input_chis)
    
    #! Load Data
    loss_data = read_1D_data(destination=data_path, name_of_file='losses.txt')
    best_vars = read_1D_data(destination=data_path, name_of_file='optimalvars.txt')
    print(f'Job {i}: Done')
    #? Just to keep the format, there is a better way
    return np.array([*best_vars,np.min(loss_data)])

#! In case one wants to run one optimizer
if __name__=="__main__":
    CONST = constants.constants
    opt = Optimizer(target_site=constants.solver_params['target'],
                    DataExist=False,
                    const=CONST,
                    Print=True)
    opt(*CONST['chis'])
