#!/usr/bin/env python3

from numpy.random import randint
import pathlib
import os
from datetime import datetime
import argparse
import multiprocessing as mp

import tet.constants
from tet.solver_mp import solver_mp
from tet.data_process import createDir

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="python3 array_solver.py")
    parser.add_argument('-p', '--path', nargs='?', type=pathlib.Path, required=True, help='Path to create data dir.')
    parser.add_argument('-c', '--constants', type=pathlib.Path, required=False, help='Path to constants json file.')
    parser.add_argument('-n', '--ncpus', default=mp.cpu_count() // 2, type=int, required=False, help='Number of cpus to use in the process pool.')

    cmd_args = parser.parse_args()
    data_path = os.path.join(cmd_args.path, f'data_{datetime.now()}')

    try:
        if os.path.exists(cmd_args.path):
            createDir(destination=data_path, replace_query=False)
        else:
            print(f"Input data path {cmd_args.path} does not exist! Creating it.")
            raise OSError()
    except OSError:
        os.makedirs(data_path)

    if cmd_args.constants != None:
        if not os.path.exists(cmd_args.constants):
            raise OSError('Provided path does not exist.')
        else:
            parameter_dict = tet.constants.loadConstants(cmd_args.constants)
    else:
        print('\n* No json file was provided, using default parameters. *\n')
        parameter_dict = {'constants': tet.constants.system_constants, 'tensorflow_params': tet.constants.TensorflowParams}

    const = {**tet.constants.system_constants, **parameter_dict['constants']}
    tet.constants.TensorflowParams = {**tet.constants.TensorflowParams, **parameter_dict['tensorflow_params']}
    tet.constants.solver_params['target'] = const['sites'] - 1

    # Set sites to optimize
    keys = [f'x{k}lims' for k in tet.constants.TensorflowParams['train_sites']]
    lims = parameter_dict['limits']
    trainable_vars_lims = dict(zip(keys, lims))

    result = solver_mp(
        const=const, 
        trainable_vars_limits=trainable_vars_lims, 
        lr=tet.constants.TensorflowParams['lr'], 
        beta_1=tet.constants.TensorflowParams['beta_1'], 
        amsgrad=tet.constants.TensorflowParams['amsgrad'],
        target_site=tet.constants.solver_params['target'],
        data_path=data_path, 
        method='bins', 
        write_data=True, 
        cpu_count=cmd_args.ncpus
    )

    print(result)
    