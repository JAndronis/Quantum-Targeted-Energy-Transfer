#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow import keras
import tet

class RLModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

max_N = 12
omegaA, omegaD = 3, -3
# chiA, chiD = -0.5, 0.5
coupling_lambda = 1
t_max = 2000
xA = np.linspace(-5, 5, 10)
xD = np.linspace(-5, 5, 10)

cwd = os.getcwd()
data = f"{cwd}/data"

coupling_dir = f"coupling-{coupling_lambda}"
coupling_dir_path = os.path.join(data, coupling_dir)

t_dir = f"tmax-{t_max}"
t_dir_path = os.path.join(coupling_dir_path, t_dir)

data_dest = os.path.join(t_dir_path, "avg_N")

tet.data_process.createDir(data)
tet.data_process.createDir(coupling_dir_path)
tet.data_process.createDir(t_dir_path)
tet.data_process.createDir(data_dest)

test_data = tet.execute(chiA=xA, 
                        chiD=xD, 
                        coupling_lambda=coupling_lambda, 
                        omegaA=omegaA, 
                        omegaD=omegaD, 
                        max_N=max_N, 
                        max_t=t_max, 
                        data_dir=os.getcwd())
# %%
