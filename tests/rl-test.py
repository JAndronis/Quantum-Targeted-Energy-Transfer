#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tet

class RLModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__=="__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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

    test_data = tet.Execute(chiA=xA, 
                            chiD=xD, 
                            coupling_lambda=coupling_lambda, 
                            omegaA=omegaA, 
                            omegaD=omegaD, 
                            max_N=max_N, 
                            max_t=t_max, 
                            data_dir=data_dest,
                            return_data=True).executeGrid()
                            
    df = pd.DataFrame(test_data)
    XA, XD = np.meshgrid(xA, xD)

    counter = 0
    test_z = np.zeros(len(xA)*len(xD))
    for i in enumerate(test_data): 
        test_z[i[0]] = min(i[1])
        counter += 1
    test_z = test_z.reshape(len(xA), len(xD))

    figure, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
    plot = ax.plot_surface(XA, XD, test_z, cmap='gnuplot')
    ax.set_xlabel('xD')
    ax.set_ylabel('xA')
    plt.show()

    plt.contourf(test_z,cmap = 'gnuplot',extent=[min(xD),max(xD),max(xA),min(xA)], levels=50)
    plt.xlabel('xD')
    plt.ylabel('xA')
    plt.colorbar()
    plt.show()

# %%
