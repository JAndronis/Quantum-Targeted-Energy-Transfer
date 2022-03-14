from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os

from tet.constants import Constants
from tet.Optimizer import Optimizer
from tet.data_process import read_1D_data
from tet.saveFig import saveFig

CONST = Constants()

if __name__=="__main__":
    NPointsxA = 4
    NPointsxD = 4
    ChiAInitials= np.linspace(-3, 3, NPointsxA)
    ChiDInitials= np.linspace(-3, 3, NPointsxD)
    Combinations = list(product(ChiAInitials, ChiDInitials))
    data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
    
    # check if data exists
    if os.path.isdir(data_path):
        if not os.listdir(data_path):
            data_exists = False
        else:    
            data_exists = True
    else:
        data_exists = False
    
    # Load Background
    min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(CONST.coupling)+'/tmax-'+
    str(CONST.max_t)+'/avg_N/min_n_combinations')
    test_array = np.loadtxt(min_n_path)
    xA_plot = test_array[:,0].reshape(CONST.plot_res,CONST.plot_res)
    xD_plot = test_array[:,1].reshape(CONST.plot_res,CONST.plot_res)
    avg_n = test_array[:,2].reshape(CONST.plot_res,CONST.plot_res)
    
    figure2, ax2 = plt.subplots(figsize=(7,7))
    plot2 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
    ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure2.colorbar(plot2)
    
    for index,(ChiAInitial, ChiDInitial) in enumerate(Combinations):
        if not data_exists:
            print('-'*20+'Combination:{} out of {}, Initial (xA,xD):({:.3f},{:.3f})'.\
                format(index, len(Combinations)-1, ChiAInitial, ChiDInitial) + '-'*20)
        opt = Optimizer(ChiAInitial=ChiAInitial, ChiDInitial=ChiDInitial,\
            DataExist=data_exists, Case = index, Plot=False)
        opt()
        
        CombinationPath = os.path.join(data_path,f'combination_{index}')
        
        # Load Data
        loss_data = read_1D_data(destination=CombinationPath, name_of_file='losses.txt')
        a = read_1D_data(destination=CombinationPath, name_of_file='xAs Trajectory.txt')
        d = read_1D_data(destination=CombinationPath, name_of_file='xDs Trajectory.txt')
        a_init = ChiAInitial
        d_init = ChiDInitial
        
        #Plot heatmaps with optimizer predictions
        # titl = f'N={CONST.max_N}, tmax={CONST.max_t}, Initial (χA, χD) = {a_init, d_init}, λ={CONST.coupling}, ωA={CONST.omegaA}, ωD={CONST.omegaD}'    
        
        x = np.array(np.array(d))
        y = np.array(np.array(a))
        ax2.plot(x, y, marker='o', color='black', label=f'Optimizer-{index} Predictions')
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
        ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, label='Initial Value', zorder=3)
        # ax2.legend(prop={'size': 15})
        # ax2.set_title(titl, fontsize=20)
        
    saveFig(fig_id="contour_final", destination=data_path)