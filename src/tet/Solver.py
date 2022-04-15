from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

from tet import constants
from Optimizer import Optimizer
from tet.data_process import createDir, read_1D_data
from tet.saveFig import saveFig

def solver(a_lims, d_lims, iterations=500, learning_rate=0.01, create_plot=False):
    const = constants.loadConstants()
    done = False
    
    # make a grid of uniformly distributed initial parameter guesses
    xa = np.linspace(a_lims[0], a_lims[1], const['resolution'])
    xd = np.linspace(d_lims[0], d_lims[1], const['resolution'])
    Combinations = list(product(xa,xd))
    
    # Init an array to save chiAs/chiDs and resulting losses
    all_losses = np.ones((len(Combinations), 3))*const['max_N']
    p = os.path.join(os.getcwd(), f'data')
    createDir(destination=p, replace_query=True)

    if create_plot:
        # Load Background
        min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(const['coupling'])+'/tmax-'+\
            str(const['max_t'])+'/avg_N/min_n_combinations')
        test_array = np.loadtxt(min_n_path)
        xA_plot = test_array[:,0].reshape(const['resolution'],const['resolution'])
        xD_plot = test_array[:,1].reshape(const['resolution'],const['resolution'])
        avg_n = test_array[:,2].reshape(const['resolution'],const['resolution'])
        
        figure2, ax2 = plt.subplots(figsize=(7,7))
        plot1 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot1)
    
    for i,(ChiAInitial, ChiDInitial) in enumerate(Combinations):
        if not done:
            # where to save the data
            data_path = os.path.join(p, f'data_{i}')
            
            opt = Optimizer(target_site='x2',
                            DataExist=False,
                            data_path=data_path,
                            const=const,
                            lr=learning_rate,
                            iterations=iterations)

            # # Read Data from the i-th combination
            # CombinationPath = os.path.join(data_path,f'combination_{i}')

            # check if data exists
            if os.path.isdir(data_path):
                if not os.listdir(data_path):
                    data_exists = False
                else:    
                    data_exists = True
            else:
                data_exists = False

            if not data_exists:
                print('-'*20+'Combination:{} out of {}, Initial (xA,xD):({:.3f},{:.3f})'.\
                    format(i, len(Combinations)-1, ChiAInitial, ChiDInitial) + '-'*20)
                
            opt(ChiAInitial, ChiDInitial)
            
            # Load Data
            loss_data = read_1D_data(destination=data_path, name_of_file='losses.txt')
            a_init = ChiAInitial
            d_init = ChiDInitial
            a = const['xA']
            d = const['xD']
            all_losses[i] = np.array([a, d, np.min(loss_data)])
        
        if create_plot:
            a = read_1D_data(destination=data_path, name_of_file='xAtrajectory.txt')
            d = read_1D_data(destination=data_path, name_of_file='xDtrajectory.txt')
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
        
        # Finish iteration if TET is found
        if np.min(loss_data)<0.1:
            done=True
            break

    if create_plot:
        # Produce legend and save plot
        ax2.legend()
        saveFig(fig_id="contour_final", destination=data_path)

    min_a, min_d = all_losses[np.argmin(all_losses[:,2]), 0], all_losses[np.argmin(all_losses[:,2]), 1]
    min_loss = all_losses[np.argmin(all_losses[:,2]), 2]
    return min_a, min_d, min_loss

if __name__=="__main__":

    # Set globals
    constants.setConstant('max_N', 3)
    constants.setConstant('max_t', 200)
    constants.setConstant('omegaA', -3)
    constants.setConstant('omegaD', 3)
    constants.setConstant('omegaMid', -3)
    constants.setConstant('coupling', 1)
    constants.setConstant('xMid', 0)
    constants.setConstant('sites', 3)
    constants.setConstant('resolution', 5)
    CONST = constants.constants
    constants.dumpConstants()

    solver(a_lims=[-5,5], d_lims=[-5,5])
    exit(0)