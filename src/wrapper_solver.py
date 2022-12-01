
if __name__=="__main__":

    from numpy.random import randint
    import os
    import shutil
    import sys
    import glob
    from os.path import exists
    import matplotlib.pyplot as plt
    from math import sqrt
    import warnings
    from datetime import datetime
    import tensorflow as tf
    import argparse

    from tet.Optimizer import Optimizer
    import tet.constants
    from tet.solver_mp import solver_mp
    from tet.data_process import createDir
    import tet.saveFig as saveFig

    parser = argparse.ArgumentParser(description="python3 array_solver.py")
    parser.add_argument('-p', '--path', nargs='?', type=pathlib.Path, required=True)
    # parser.add_argument('--id', type=int)
    # parser.add_argument('--array-size', type=int)

    cmd_args = parser.parse_args()

    try:
        data_path = os.path.join(cmd_args.path, f'data_{datetime.now()}_{cmd_args.id}')
        if os.path.exists(cmd_args.path):
            createDir(destination=data_path, replace_query=False)
        else:
            print(f"Input data path {cmd_args.path} does not exist! Creating it.")
            raise OSError()
    except OSError:
        os.makedirs(data_path)

    const = {
        "max_N": 1,
        "max_t": 25,
        "omegas": [
            -3,
            3
        ],
        "chis": [
            0,
            0
        ],
        "coupling": 0.1,
        "sites": 2,
        "timesteps": 30,
        "min_n": 1
    }

    tet.constants.TensorflowParams['train_sites'] = [0, 1]
    keys = [f'x{k}lims' for k in tet.constante.TensorflowParams['train_sites']] 
    lims = [[randint(-10,0), randint(0,10)], [randint(-10,0), randint(0,10)]]
    TrainableVarsLimits = dict(zip(keys,lims))

    solver_mp(
        const=const_iter, TrainableVarsLimits=TrainableVarsLimits, lr=0.1, beta_1=0.4, amsgrad=True,
        data_path=data_path, method='bins', return_values=True, write_data=True
    )
    