#!/usr/bin/env python3

import numpy as np
import time
import sys
import argparse
import pathlib
import os

from tet import constants
from tet.Optimizer import mp_opt, Optimizer
from tet.data_process import createDir, read_1D_data
from tet.constants import solver_params, TensorflowParams
from tet.solver_mp import getCombinations

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="python3 array_solver.py")
    parser.add_argument('-p','--path', nargs='?', type=pathlib.Path, required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--array-size', type=int, required=True)

    cmd_args = parser.parse_args()

    try:
        data_path = os.path.join(cmd_args.path, f'data_{time.time_ns()}_{cmd_args.id}')
        if os.path.exists(cmd_args.path):
            createDir(destination=data_path, replace_query=True)
        else:
            print(f"Input data path {cmd_args.path} does not exist! Creating it.")
            raise OSError()
    except OSError:
        os.makedirs(data_path)

    # Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Initialize helper parameters
    const = constants.constants.copy()
    TrainableVarsLimits = {'x1lims': [-40, 40]}
    lims = list(TrainableVarsLimits.values())
    method = 'bins'
    grid = 4
    epochs_bins = 1000
    target_site = constants.acceptor

    # Control how many times loss is lower than the threshold having changed the limits
    iteration = 0

    # An array to save the optimal parameters
    OptimalVars, min_loss = np.zeros(len(TrainableVarsLimits)), const['max_N']

    Combinations = getCombinations(TrainableVarsLimits, method=method, grid=grid)
    if method=='bins': iterations = epochs_bins
    else: iterations = epochs_grid
    
    step = 9*9 // cmd_args.array_size
    for ij in range(cmd_args.id*step, (cmd_args.id+1)*step):
        if ij//9==0 or ij%9==0: continue
        const_copy = const.copy()
        const_copy['max_N'] = ij // 9
        const_copy['omegas'][0] = ij % 9
        const_copy['omegas'][1] = const_copy['omegas'][0]
        const_copy['omegas'][-1] = -const_copy['omegas'][0]
        xd = (const_copy['omegas'][-1] - const_copy['omegas'][0])/const_copy['max_N']
        xa = -xd
        const_copy['chis'] = [xd, 0, xa]
        _all_losses = []
        for i, (combination) in enumerate(Combinations):
            tmp = [i+cmd_args.id, combination, data_path, const_copy, target_site, iterations]
            _all_losses.append(mp_opt(*tmp))
            # break
        all_losses = np.array(_all_losses)
        OptimalVars = [
            float(all_losses[np.argmin(all_losses[:,const['sites']]), i]) \
            for i in range(const['sites'])
        ]
        min_loss = float(all_losses[np.argmin(all_losses[:,const['sites']]), const['sites']])
        # Print results of run
        print(f"Best parameters of tries: loss={min_loss}, OptimalVars = {OptimalVars}")
        const_copy['chis'] = OptimalVars
        const_copy['min_n'] = min_loss
        constants.dumpConstants(
            dict=const_copy, path=data_path, 
            name=f'constants_n={const_copy["max_N"]}_o={const_copy["omegas"][1]}'
        )
        # break
