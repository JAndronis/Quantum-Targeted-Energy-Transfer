#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import time
import sys
import os

import constants
from Optimizer import mp_opt, Optimizer
from data_process import createDir, read_1D_data
from constants import solver_params, TensorflowParams
from solver_mp import getCombinations

if __name__=="__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank==0:
        if len(sys.argv)<=1:
            raise OSError("A path to the data directory is needed as an call argument.")
        fn = sys.argv[1]
        try:
            data_path = os.path.join(fn, f'data_{time.time_ns()}')
            if os.path.exists(fn):
                createDir(destination=data_path, replace_query=True)
            else:
                print(f"Input data path {fn} does not exist! Creating it.")
                raise OSError()
        except OSError:
            os.makedirs(data_path)
    else:
        data_path = None

    data_path = comm.bcast(data_path, root=0)

    # Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Initialize helper parameters
    const = constants.constants.copy()
    TrainableVarsLimits = {'x1lims': [-40, 40]}
    lims = list(TrainableVarsLimits.values()) 
    _edge =  [4.0,3.5,3.0,2.5,1.5,0.5,0.1]
    method = 'bins'
    grid = 4
    epochs_bins = 1000
    target_site = constants.acceptor

    # Control how many times loss is lower than the threshold having changed the limits
    iteration = 0

    # Count attempts based on the limits 
    counter = 0
    done = False
    bin_choice = False

    # An array to save the optimal parameters
    OptimalVars, min_loss = np.zeros(len(TrainableVarsLimits)), const['max_N']

    Combinations = getCombinations(TrainableVarsLimits, method=method, grid=grid)
    if method=='bins': iterations = epochs_bins
    else: iterations = epochs_grid
    
    perrank = 9*9 // size
    for ij in range(rank*perrank, (rank+1)*perrank):
        if ij//9==0 or ij%9==0: continue
        const_copy = const.copy()
        const_copy['max_N'] = ij // 9
        const_copy['omegas'][0] = ij % 9
        const_copy['omegas'][1] = -const_copy['omegas'][0]
        const_copy['omegas'][-1] = -const_copy['omegas'][0]
        xd = (const_copy['omegas'][-1] - const_copy['omegas'][0])/const_copy['max_N']
        xa = -xd
        const_copy['chis'] = [xd, 0, xa]
        # print(f'Rank {rank} working on parameters: {const_copy}.')
        _all_losses = []
        for i, (combination) in enumerate(Combinations):
            tmp = [i+rank, combination, data_path, const_copy, target_site, iterations]
            _all_losses.append(mp_opt(*tmp))
        all_losses = np.array(_all_losses)
        OptimalVars = [
            float(all_losses[np.argmin(all_losses[:,const['sites']]), i]) \
            for i in range(const['sites'])
        ]
        min_loss = float(all_losses[np.argmin(all_losses[:,const['sites']]), const['sites']])
        # Print results of run
        print(f"Best parameters of tries: loss={min_loss}, OptimalVars = {OptimalVars}")
        if min_loss<=const_copy["max_N"]/2:
            counter += 1
            edge = _edge[iteration]
            #grid += 2
            lims = [[OptimalVars[i]-edge, OptimalVars[i]+edge] for i in range(len(TrainableVarsLimits))]
        const_copy['chis'] = OptimalVars
        const_copy['min_n'] = min_loss
        constants.dumpConstants(
            dict=const_copy, path=data_path, 
            name=f'constants_n={const["max_N"]}_o={const["omegas"][1]}'
        )
