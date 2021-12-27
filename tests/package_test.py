import sys
from typing import Type
assert sys.version_info >= (3,6)
import os
import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
import time
import itertools
import multiprocessing as mp
import pandas as pd
from functools import partial
import warnings

from tet import execute, saveFig

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=np.ComplexWarning)

    max_N = 12
    omegaA, omegaD = 3, -3
    chiA, chiD = -0.5, 0.5
    coupling_lambda = 0.001
    max_t = 2000
    # xA = np.linspace(-4, 4, 100)
    # xD = np.linspace(-4, 4, 100)

    cwd = os.getcwd()
    new_data = f"{cwd}/new_data"

    try:
        os.mkdir(new_data)
    except OSError as error:
        print(error)
        while True:
            query = input("Directory exists, replace it? [y/n] ")
            fl_1 = query[0].lower() 
            if query == '' or not fl_1 in ['y','n']: 
                print('Please answer with yes or no')
            else: break
        if fl_1 == 'n': sys.exit(0)

    dir_name = f"coupling-{coupling_lambda}"
    data_dest = os.path.join(new_data, dir_name)
    print("Writing values in:", data_dest)
    try:
        os.mkdir(data_dest)
    except OSError as error:
        print(error)
        while True:
            query = input("Directory exists, replace it? [y/n] ")
            fl_2 = query[0].lower() 
            if query == '' or not fl_2 in ['y','n']: 
                print('Please answer with yes or no')
            else: break
        if fl_2 == 'n': sys.exit(0)

    t1 = time.time()
    execute.execute(chiA, chiD, coupling_lambda, omegaA, omegaD, max_N, data_dest)
    t2 = time.time()
    dt = t2-t1
    print(f"Code took: {dt:.3f}secs to run")

    df = pd.read_csv(os.path.join(data_dest, os.listdir(data_dest)[0]))
    df.plot()
    saveFig.saveFig("average_number_of_bosons")
    print()