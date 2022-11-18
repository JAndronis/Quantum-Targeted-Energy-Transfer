
if __name__=="__main__":

    import numpy as np
    import os
    import shutil
    import sys
    import glob
    from os.path import exists
    import matplotlib.pyplot as plt
    from math import sqrt
    import warnings
    import tensorflow as tf

    from tet.Optimizer import Optimizer
    import tet.constants
    from tet.solver_mp import solver_mp
    from tet.data_process import createDir
    import tet.saveFig as saveFig

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data_path_cwd = '/home/ph4783/data/data_figure_3_n4_last'
    createDir(data_path_cwd, replace_query=False)

    for i in [4]:

        print(f'\nComputing results for N = {i}\n')

        const = {
            "max_N": i,
            "max_t": 25,
            "omegas": [
                -3,
                3
            ],
            "chis": [
                -6.0,
                6.0
            ],
            "coupling": 0.1,
            "sites": 2,
            "timesteps": 30,
            "min_n": 1
        }

        tet.constants.TensorflowParams['train_sites'] = [0, 1]
        keys = [f'x{i}lims' for i in tet.constants.TensorflowParams['train_sites']] 
        lims = [[-10,10]]*len(keys)
        TrainableVarsLimits = dict(zip(keys,lims))

        tet.constants.solver_params['Npoints'] = 15

        data_path_N = os.path.join(data_path_cwd, f'data_N{i}')

        opt = solver_mp(
            TrainableVarsLimits=TrainableVarsLimits, const=const, 
            data_path=data_path_N, method='grid', lr=0.1, beta_1=0.9, amsgrad=False
        )
