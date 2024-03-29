import sys
import os
import time
import numpy as np
import tensorflow as tf
import keras.backend as K

from .data_process import createDir, writeData
from .HamiltonianLoss import Loss
from .constants import TensorflowParams

assert tf.__version__ >= "2.0"
assert sys.version_info >= (3, 6)


# from solver_mp_test import getCombinations
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class Optimizer:
    """
    A class instance of a keras optimizer, along with relevant parameters.
    
    Args:
        const (dict): Refer to the system_constants dictionary in constants.py.
        target_site (int): Refer to the argument target of the solver_params dictionary in constants.py
        DataExist (bool, optional): A boolean variable verifying that an optimizer with given initial guesses runs for the first time. Defaults to False.
        Print (bool, optional): Parameter defining if to print results of optimization on the console. Defaults to True.
        iterations (int, optional): Refer to the argument iterations of the TensorflowParams dictionary in constants.py. Defaults to the value from constants.py.
        opt (tf.keras.optimizers, optional): A tensorflow optimizer which is going to be used for the minimization of the HamiltonianLoss. Defaults to tf.keras.optimizers.Adam().
        data_path (str, optional): Path to save the directory of an optimizer with given initial guesses. Defaults to os.path.join(os.getcwd(), 'data_optimizer').
    """

    def __init__(
            self, const: dict, target_site: int,
            DataExist=False, Print=True,
            iterations=TensorflowParams['iterations'],
            train_sites=TensorflowParams['train_sites'],
            opt=tf.keras.optimizers.Adam(),
            data_path=os.path.join(os.getcwd(), 'data_optimizer')
    ):

        # ! Import the parameters of the problem
        self.const = const

        self.coupling = self.const['coupling']
        self.max_t = self.const['max_t']
        self.max_n = self.const['max_N']
        self.omegas = self.const['omegas']
        self.sites = self.const['sites']

        self.train_sites = train_sites
        self.target_site = target_site
        self.DataExist = DataExist
        self.data_path = data_path
        self.iter = iterations
        self.opt = opt
        self.lr = opt.learning_rate
        self.Print = Print

        self.vars = [None for _ in range(len(self.const['chis']))]

        self.DTYPE = tf.float64

    def __call__(self, *args, write_data=False):
        self.init_chis = list(args)

        if self.DataExist:
            pass
        else:
            if write_data:
                # opt_path = os.path.join(self.data_path, 'data_optimizer')
                createDir(self.data_path, replace_query=False)
                self._train(write_data=True)
                return self.results
            else:
                self._train(write_data)
                return self.results

    # Compute the loss function
    @tf.function(jit_compile=False)
    def compute_loss(self):
        return self.loss(self.vars, site=self.target_site)

    # Get the gradients
    def get_grads(self):
        with tf.GradientTape() as t:
            # t watches the trainable parameters only by default
            loss = self.compute_loss()
        grads = t.gradient(loss, self.vars)
        del t
        return grads, loss

    # Apply the gradients
    @tf.function(jit_compile=False)
    def apply_grads(self, grads):
        self.opt.apply_gradients(zip(grads, self.vars))

    # Running the optimizer given initial guesses for the trainable parameters
    def train(self, initial_chis):
        # * Reset Optimizer
        K.clear_session()
        for var in self.opt.variables():
            var.assign(tf.zeros_like(var))
        K.set_value(self.opt.learning_rate, self.lr)

        # * Define the object of the Loss class according to the current trainable parameters.
        # ? ATTENTION
        self.loss = Loss(const=self.const)

        # Save the values of the loss function while proceeding 
        mylosses = []

        # Define the tolerance of the optimizer
        self.tol = TensorflowParams['tol']

        # Discriminate trainable and non-trainable parameters
        for i in range(len(self.const['chis'])):
            if self.vars[i] is None:
                # Trainable ones
                if i in self.train_sites:
                    self.vars[i] = tf.Variable(
                        initial_value=initial_chis[i], dtype=self.DTYPE,
                        name=f'chi{i}', trainable=True
                    )
                # Non-trainable ones
                else:
                    self.vars[i] = tf.Variable(
                        initial_value=initial_chis[i], dtype=self.DTYPE,
                        name=f'chi{i}', trainable=False
                    )

        # Nonlinearity parameters that produce the lowest loss function
        best_vars = [tf.Variable(initial_value=0, dtype=self.DTYPE, trainable=False) for _ in range(len(self.vars))]
        # Add the initial value of the loss function to ensure that following condition statements will apply
        mylosses.append(self.max_n)

        best_loss = self.max_n
        current_loss = best_loss
        # Keep the changes for each parameter. Non-trainable parameters are not supposed to change
        counter = 0
        var_data = [[] for _ in range(len(self.vars))]
        # A help list for controlling the interruption of the iterative procedure
        var_error_count = [0 for _ in range(len(self.vars))]

        t0 = time.time()
        for epoch in range(self.iter):

            # Help list with the variables before applying gradients
            _vars = [self.vars[i].numpy() for i in range(len(self.vars))]

            # Compute the loss function and obtain the updated variables
            grads, loss = self.get_grads()

            # Set a repetition rate for displaying the progress 
            if self.Print:
                if epoch % 50 == 0:
                    print(f'Loss:{loss.numpy()}, ', *[f'x{j}: {self.vars[j].numpy()}, ' for j in range(len(self.vars))],
                          f', epoch:{epoch}')

            # Learning Rate Scheduling
            # if epoch%200==0:
            #     K.set_value(self.opt.learning_rate, self.opt.learning_rate/(0.01*epoch))

            # Reduce the learning rate when being close to TET
            if loss.numpy() <= 0.1:
                K.set_value(self.opt.learning_rate, 0.0001)

            # Save the new value of the loss function
            mylosses.append(loss.numpy())

            # Keep the minimum loss and the corresponding parameters
            if mylosses[epoch + 1] < min(list(mylosses[:epoch + 1])):
                for i in range(len(self.vars)):
                    best_vars[i].assign(self.vars[i].numpy())
                best_loss = mylosses[epoch + 1]

            # CHANGE self.vars
            self.apply_grads(grads)

            # Manual check for the progress of each variable
            var_error = [np.abs(self.vars[i].numpy() - _vars[i]) for i in range(len(self.vars))]

            # Change self.vars to 1 step back if loss is reduced too fast
            if mylosses[epoch + 1] - mylosses[epoch] >= 0.5 and loss.numpy() >= 0.5:
                for i in range(len(self.vars)):
                    self.vars[i].assign(_vars[i])

            # Keep the changes of the variables per 10 steps for plotting
            counter += 1
            if counter % 10 == 0:
                for k in range(len(self.vars)):
                    var_data[k].append(self.vars[k].numpy())

            # Interrupt in case of TET 
            if np.abs(loss.numpy()) < 0.1:
                break

            t1 = time.time()
            dt = t1 - t0

            # Interrupt in case of non-progress
            for j in range(len(self.vars)):
                # Non-trainable parameters change slightly
                if var_error[j] < self.tol and j in self.train_sites:
                    var_error_count[j] += 1
                    if var_error_count[j] > 2:
                        if self.Print:
                            print(f'Stopped training because of x{j}_new-x{j}_old =', var_error[j])
                            print(
                                *[f"\nApproximate value of chi_{j}: {best_vars[j].numpy()}" for j in
                                  range(len(self.vars))],
                                "\nLoss:", best_loss,
                                "\nOptimizer Iterations:", self.opt.iterations.numpy(),
                                "\nTraining Time:", dt,
                                "\n" + 60 * "-",
                                "\nParameters:",
                                "\nOmegas", self.omegas,
                                "| N:", self.max_n,
                                "| Sites: ", self.sites,
                                "| Total timesteps:", self.max_t,
                                "| Coupling Lambda:", self.coupling,
                                "\n" + 60 * "-"
                            )
                        return mylosses, var_data, best_vars

        # Print the outcome of the optimizer
        if self.Print:
            print(
                *[f"\nApproximate value of chi_{j}: {best_vars[j].numpy()}" for j in range(len(self.vars))],
                "\nLoss:", best_loss,
                "\nOptimizer Iterations:", self.opt.iterations.numpy(),
                "\nTraining Time:", dt,
                "\n" + 60 * "-",
                "\nParameters:",
                "\nOmegas", self.omegas,
                "| N:", self.max_n,
                "| Sites: ", self.sites,
                "| Total timesteps:", self.max_t,
                "| Coupling Lambda:", self.coupling,
                "\n" + 60 * "-"
            )
        return mylosses, var_data, best_vars

    # ! Run the optimizer for given initial guesses and save trajectories.
    def _train(self, write_data: bool) -> any:
        mylosses, var_data, best_vars = self.train(self.init_chis)

        if write_data:
            # Save the evolution of the values of the loss function
            writeData(data=mylosses[1:], destination=self.data_path, name_of_file='losses.txt')

            # Save initial parameter data
            writeData(data=self.init_chis, destination=self.data_path, name_of_file='init_chis.txt')

            for i in range(len(var_data)):
                # Save the trajectories in a file
                writeData(data=var_data[i], destination=self.data_path, name_of_file=f'x{i}trajectory.txt')

        self.results = {
            'loss': mylosses[1:],
            'var_data': var_data,
            'best_vars': [i.numpy() for i in best_vars],
        }


# ----------------------------- Multiprocess Helper Function ----------------------------- #

def mp_opt(
        i: int, combination: list, iteration_path: str,
        const: dict, target_site: int, iterations: int,
        lr: float, beta_1: float, amsgrad_bool: bool,
        write_data: bool, train_sites: list
) -> np.ndarray:
    """
    A helper function used for multiprocess.
    
    Args:
        lr:
        beta_1:
        amsgrad_bool:
        train_sites:
        write_data:
        i (int): Index referring to optimizer with specific initial guesses
        combination (list): The initial guesses(referring to the trainable parameters) of the said optimizer
        iteration_path (str): Path of the iteration directory.
        const (dict): Refer to the constants dictionary in constants.py.
        target_site(int): Refer to the argument target of the solver_params dictionary in constants.py
        iterations(int): Maximum iterations of the optimizer
    """

    # ! Import the parameters of the problem
    data_path = os.path.join(iteration_path, f'data_optimizer_{i}')

    # ! Create the current optimizer
    opt = Optimizer(
        target_site=target_site,
        DataExist=False,
        Print=False,
        data_path=data_path,
        train_sites=train_sites,
        const=const,
        opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, amsgrad=amsgrad_bool),
        iterations=iterations
    )

    # ! Call the optimizer with chis including the given initial guesses
    input_chis = [0] * len(const['chis'])

    # Update the list with the initial guesses of the optimizer. IT IS ESSENTIAL WHEN WE DON'T TRAIN ALL THE NON 
    # LINEARITY PARAMETERS
    for index, case in zip(train_sites, combination):
        input_chis[index] = case

    results = opt(*input_chis, write_data=write_data)

    # ! Load Data
    loss_data = results['loss']
    best_vars = results['best_vars']
    print(f'Job {i}: Done')

    return np.array([*best_vars, np.min(loss_data)])
