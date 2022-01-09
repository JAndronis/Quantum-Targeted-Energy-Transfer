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

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    
    max_N = 12
    omegaA, omegaD = 2, -3
    chiA, chiD = -0.5, 0.5
    coupling_lambda = 0.001
    t_max = 2000

    xA = np.linspace(-6, 6, 100)
    xD = np.linspace(-6, 6, 100)

    return_query = True

    cwd = os.getcwd()
    data = f"{cwd}/data"

    coupling_dir = f"coupling-{coupling_lambda}"
    coupling_dir_path = os.path.join(data, coupling_dir)

    t_dir = f"tmax-{t_max}"
    t_dir_path = os.path.join(coupling_dir_path, t_dir)

    data_dest = os.path.join(t_dir_path, "avg_N")

    tet.data_process.createDir(data, replace=False)
    tet.data_process.createDir(coupling_dir_path, replace=False)
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
                            return_data=return_query).executeGrid()

    if return_query:                        
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
        titl = f'tmax = {t_max}, points χA, χD = {len(xA), len(xD)}, λ={coupling_lambda}, ωA={omegaA}, ωD={omegaD}'
        ax.set_title(titl)
        tet.saveFig(titl+' - 3dplot', t_dir_path)

        figure2, ax2 = plt.subplots(figsize=(12,12))
        plot2 = ax2.contourf(test_z,cmap = 'gnuplot',extent=[min(xD),max(xD),max(xA),min(xA)], levels=50)
        ax2.set_xlabel('xD')
        ax2.set_ylabel('xA')
        figure2.colorbar(plot2)
        ax2.set_title(titl)
        tet.saveFig(titl+' - contourplot', t_dir_path)

if __name__=="__main__":
    main()