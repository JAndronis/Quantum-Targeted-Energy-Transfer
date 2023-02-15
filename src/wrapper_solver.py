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
            raise OSError('Provided constants path does not exist.')
        else:
            const = tet.constants.loadConstants(cmd_args.constants)
    else:
        print('\n*No json file was provided, using default constants.*\n')
        const = tet.constants.system_constants

    # Set sites to optimize
    tet.constants.TensorflowParams['train_sites'] = [0, 1]
    keys = [f'x{k}lims' for k in tet.constants.TensorflowParams['train_sites']]
    lims = [[randint(-10, 0), randint(0, 10)], [randint(-10, 0), randint(0, 10)]]
    trainable_vars_lims = dict(zip(keys, lims))

    result = solver_mp(
        const=const, trainable_vars_limits=trainable_vars_lims, lr=0.1, beta_1=0.4, amsgrad=True,
        data_path=data_path, method='bins', write_data=True, cpu_count=cmd_args.ncpus
    )

    print(result)
    