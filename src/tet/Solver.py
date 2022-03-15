from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os

from tet.constants import Constants
from Optimizer import Optimizer
from tet.data_process import read_1D_data
from tet.saveFig import saveFig

# Constants
CONST = Constants()

def solver(xmin_a, xmax_a, xmin_d, xmax_d, grid_size, case, iterations=500, learning_rate=0.01, create_plot=False):
    NPointsxA = grid_size
    NPointsxD = grid_size
    ChiAInitials= np.linspace(xmin_a, xmax_a, NPointsxA)
    ChiDInitials= np.linspace(xmin_d, xmax_d, NPointsxD)
    Combinations = list(product(ChiAInitials, ChiDInitials))
    data_path = os.path.join(os.getcwd(), f'data_optimizer_avgn_{case}')
    
    # check if data exists
    if os.path.isdir(data_path):
        if not os.listdir(data_path):
            data_exists = False
        else:    
            data_exists = True
    else:
        data_exists = False
    
    # Init an array to save initial chiAs/chiDs and resulting losses
    all_losses = np.zeros((len(Combinations), 3))
    
    if create_plot:
        # Load Background
        min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(CONST.coupling)+'/tmax-'+
        str(CONST.max_t)+'/avg_N/min_n_combinations')
        test_array = np.loadtxt(min_n_path)
        xA_plot = test_array[:,0].reshape(CONST.plot_res,CONST.plot_res)
        xD_plot = test_array[:,1].reshape(CONST.plot_res,CONST.plot_res)
        avg_n = test_array[:,2].reshape(CONST.plot_res,CONST.plot_res)
        
        figure2, ax2 = plt.subplots(figsize=(7,7))
        plot1 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot1)
    
    for i,(ChiAInitial, ChiDInitial) in enumerate(Combinations):
        if not data_exists:
            print('-'*20+'Combination:{} out of {}, Initial (xA,xD):({:.3f},{:.3f})'.\
                format(i, len(Combinations)-1, ChiAInitial, ChiDInitial) + '-'*20)
            
        opt = Optimizer(ChiAInitial=ChiAInitial,
                        ChiDInitial=ChiDInitial,
                        DataExist=data_exists, 
                        data_path=data_path,
                        Case = i, 
                        Plot=False,
                        lr=learning_rate,
                        iterations=iterations)
        opt()
        
        # Read Data from the i-th combination
        CombinationPath = os.path.join(data_path,f'combination_{i}')
        
        # Load Data
        loss_data = read_1D_data(destination=CombinationPath, name_of_file='losses.txt')
        a_init = ChiAInitial
        d_init = ChiDInitial
        all_losses[i] = np.array([a_init, d_init, np.min(loss_data)])
        if create_plot:
            a = read_1D_data(destination=CombinationPath, name_of_file='xAtrajectory.txt')
            d = read_1D_data(destination=CombinationPath, name_of_file='xDtrajectory.txt')
        
        if create_plot:
            # Plot trajectory and initial guess data
            x = np.array(np.array(d))
            y = np.array(np.array(a))
            plot2 = ax2.plot(x, y, marker='o', color='black', label=f'Optimizer Predictions' if i == 0 else "")
            u = np.diff(x)
            v = np.diff(y)
            pos_x = x[:-1] + u/2
            pos_y = y[:-1] + v/2
            norm = np.sqrt(u**2+v**2)
            ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
            plot3 = ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, label='Initial Guess' if i == 0 else "", zorder=3)
            
    if create_plot:
        # Produce legend and save plot
        ax2.legend()    
        saveFig(fig_id="contour_final", destination=data_path)
    
    min_a_init, min_d_init = all_losses[np.argmin(all_losses[:,2]), 0], all_losses[np.argmin(all_losses[:,2]), 1]
    min_loss = all_losses[np.argmin(all_losses[:,2]), 2]
    return min_a_init, min_d_init, min_loss

if __name__=="__main__":
    # Make an initial search of the parameter space
    min_a, min_d, loss = solver(xmin_a=-3, xmax_a=3, xmin_d=-3, xmax_d=3,\
        grid_size=2, case=0, iterations=500, learning_rate=0.1, create_plot=True)
    
    a_min, a_max = min_a-1, min_a+1
    d_min, d_max = min_d-1, min_d+1
    xmin, xmax, loss = solver(xmin_a=a_min, xmax_a=a_max, xmin_d=d_min, xmax_d=d_max,\
        grid_size=4, case=1, iterations=1000, learning_rate=0.01, create_plot=True)