import tet.constants as constants
import tet.saveFig as saveFig
import numpy as np
import os
import shutil
import sys
import glob
from os.path import exists
import matplotlib.pyplot as plt
from math import sqrt
import warnings

from tet.Optimizer import Optimizer
import tet.constants
from tet.solver_mp import solver_mp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
const = {
    "max_N": 3,
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

keys = [f'x{i}lims' for i in tet.constants.TensorflowParams['train_sites']] 
lims = [[-10,10]]*len(keys)
TrainableVarsLimits = dict(zip(keys,lims))

tet.constants.solver_params['Npoints'] = 20

opt = solver_mp(TrainableVarsLimits=TrainableVarsLimits, const=const, data_path='/home/ph4783/data/data', method='grid')

