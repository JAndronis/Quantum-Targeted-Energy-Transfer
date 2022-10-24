#!/usr/bin/env python3

import argparse
import os
import pathlib
import sys
import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from tet.constants import (TensorflowParams, acceptor, dumpConstants,
                           solver_params, system_constants)
from tet.data_process import createDir
from tet.HamiltonianLoss import Loss
from tet.Optimizer import Optimizer, mp_opt
from tet.solver_mp import getCombinations


class MPI_Optimizer(Optimizer):
    def __init__(
        self, const: dict, target_site: int,
        rank: int, size: int, threshold: float,
        DataExist=False, Print=True,
        iterations=TensorflowParams['iterations'],
        opt=tf.keras.optimizers.Adam(),
        data_path=os.path.join(os.getcwd(), 'data_optimizer')
    ):
        super().__init__(
            const, target_site,
            DataExist=DataExist, Print=Print,
            iterations=iterations,
            opt=opt,
            data_path=data_path
        )
        self.rank = rank
        self.size = size
        self.threshold = threshold

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
                if i in TensorflowParams['train_sites']:
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

        # Non linearity parameters that produce the lowest loss function
        best_vars = [
            tf.Variable(initial_value=0, dtype=self.DTYPE, trainable=False) for _ in range(len(self.vars))
        ]
        # Add the initial value of the loss function to ensure that following condition statements will apply
        mylosses.append(self.max_n)

        best_loss = self.max_n
        current_loss = best_loss
        # Keep the changes for each parameters. Non-trainable parameters are not supposed to change
        counter = 0
        var_data = [[] for _ in range(len(self.vars))]
        # A help list for controlling the interuption of the iterarive procedure
        var_error_count = [0 for _ in range(len(self.vars))]

        t0 = time.time()

        epoch = 0
        done = False
        while epoch < self.iter and done is False:

            # Help list with the variables before applying gradients
            _vars = [self.vars[i].numpy() for i in range(len(self.vars))]

            # Compute the loss function and obtain the updated variables
            grads, loss = self.get_grads()

            # Set a repetition rate for displaying the progress
            if self.Print:
                if epoch % 50 == 0:
                    print(f'Worker {self.rank} - Loss:{loss.numpy()}, ', *[f'x{j}: {self.vars[j].numpy()}, ' for j in range(
                        len(self.vars))], f', epoch:{epoch}')

            # Learning Rate Scheduling
            # if epoch%200==0:
            #     K.set_value(self.opt.learning_rate, self.opt.learning_rate/(0.01*epoch))

            # Reduce the learning rate when being close to TET
            if loss.numpy() <= 0.1:
                K.set_value(self.opt.learning_rate, 0.0001)

            # Save the new value of the loss function
            mylosses.append(loss.numpy())

            # Keep the minimum loss and the corresponding parameters
            if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
                for i in range(len(self.vars)):
                    best_vars[i].assign(self.vars[i].numpy())
                best_loss = mylosses[epoch+1]

            # CHANGE self.vars
            self.apply_grads(grads)

            # Manual check for the progress of each variable
            var_error = [np.abs(self.vars[i].numpy() - _vars[i])
                         for i in range(len(self.vars))]

            # Change self.vars to 1 step back if loss is reduced too fast
            if mylosses[epoch+1] - mylosses[epoch] >= 0.5 and loss.numpy() >= 0.5:
                for i in range(len(self.vars)):
                    self.vars[i].assign(_vars[i])

            # Keep the changes of the variables per 10 steps for plotting
            counter += 1
            if counter % 10 == 0:
                for k in range(len(self.vars)):
                    var_data[k].append(self.vars[k].numpy())

            answer = False
            # Interrupt in case of TET
            if loss.numpy() < self.threshold:
                print(f'Worker {self.rank} - found TET stopping all workers!')
                answer = comm.bcast(True, root=self.rank)
                break
            done = answer

            # Interrupt in case of non-progress
            for j in range(len(self.vars)):
                # Non trainable parameters change slightly
                if var_error[j] < self.tol and j in TensorflowParams['train_sites']:
                    var_error_count[j] += 1
                    if var_error_count[j] > 2:
                        if self.Print:
                            print(
                                f'Stopped training because of x{j}_new-x{j}_old =', var_error[j]
                            )
                        t1 = time.time()
                        dt = t1-t0
                        if self.Print:
                            print(
                                *[f"\nApproximate value of chi_{j}: {best_vars[j].numpy()}" for j in range(len(self.vars))],
                                "\nLoss:", best_loss,
                                "\nOptimizer Iterations:", self.opt.iterations.numpy(),
                                "\nTraining Time:", dt,
                                "\n"+60*"-",
                                "\nParameters:",
                                "\nOmegas", self.omegas,
                                "| N:", self.max_n,
                                "| Sites: ", self.sites,
                                "| Total timesteps:", self.max_t,
                                "| Coupling Lambda:", self.coupling,
                                "\n"+60*"-"
                            )
                        return mylosses, var_data, best_vars
            epoch += 1

        t1 = time.time()
        dt = t1-t0

        # Print the outcome of the optimizer
        if self.Print:
            print(
                *[f"\nApproximate value of chi_{j}: {best_vars[j].numpy()}" for j in range(len(self.vars))],
                "\nLoss:", best_loss,
                "\nOptimizer Iterations:", self.opt.iterations.numpy(),
                "\nTraining Time:", dt,
                "\n"+60*"-",
                "\nParameters:",
                "\nOmegas", self.omegas,
                "| N:", self.max_n,
                "| Sites: ", self.sites,
                "| Total timesteps:", self.max_t,
                "| Coupling Lambda:", self.coupling,
                "\n"+60*"-"
            )
        return mylosses, var_data, best_vars


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="python3 array_solver.py")
    parser.add_argument(
        '-p', '--path', nargs='?',
        type=pathlib.Path, required=True
    )

    cmd_args = parser.parse_args()

    try:
        data_path = os.path.join(
            cmd_args.path, f'data_{time.time_ns()}_{rank}'
        )
        if os.path.exists(cmd_args.path):
            createDir(destination=data_path, replace_query=True)
        else:
            print(
                f"Input data path {cmd_args.path} does not exist! Creating it."
            )
            raise OSError()
    except OSError:
        os.makedirs(data_path)

    # Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Initialize helper parameters
    const = system_constants.copy()
    TrainableVarsLimits = {'x1lims': [-40, 40]}
    lims = list(TrainableVarsLimits.values())
    method = 'bins'
    grid = 4
    epochs_bins = 1000

    # Control how many times loss is lower than the threshold having changed the limits
    iteration = 0

    # An array to save the optimal parameters
    OptimalVars, min_loss = np.zeros(len(TrainableVarsLimits)), const['max_N']

    Combinations = getCombinations(
        TrainableVarsLimits, method=method, grid=grid)
    if method == 'bins':
        iterations = epochs_bins
    else:
        iterations = epochs_grid

    xd = (const['omegas'][-1] -
          const['omegas'][0]) / const['max_N']
    xa = -xd
    const['chis'] = [xd, 0, xa]
    combination = Combinations[rank]

    # Update the list with the initial guesses of the optimizer. IT IS ESSENTIAL WHEN WE DON'T TRAIN ALL THE NON
    # LINEARITY PARAMETERS
    for index, case in zip(TensorflowParams['train_sites'], combination):
        const['chis'][index] = case

    opt = MPI_Optimizer(
        rank=rank,
        size=size,
        target_site=acceptor,
        DataExist=False,
        Print=True,
        data_path=data_path,
        const=const,
        opt=tf.keras.optimizers.Adam(
            learning_rate=0.5, beta_1=0.4, amsgrad=True
        ),
        iterations=iterations,
        threshold=0.15
    )

    results = opt(*const['chis'])

    const['chis'] = results['best_vars']
    const['min_n'] = min(results['loss'])
    dumpConstants(
        dict=const, path=data_path,
        name=f'constants_{rank}'
    )
    sys.exit(0)
