from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os

import constants
from Optimizer import Optimizer
from tet.data_process import read_1D_data
from tet.saveFig import saveFig

def solver(a_lims, d_lims, grid_size, case, iterations=500, learning_rate=0.01, create_plot=False):
    const = constants.loadConstants()
    done = False
    # make random initial guesses according to the number of bins
    xa = np.linspace(a_lims[0], a_lims[1], const['resolution'])
    xd = np.linspace(d_lims[0], d_lims[1], const['resolution'])
    data = np.array((list(product(xa,xd))))
    extenti = (a_lims[0]-0.1, a_lims[1]+0.1)
    extentj = (d_lims[0]-0.1, d_lims[1]+0.1)
    bins = grid_size
    _, *edges = np.histogram2d(data[:,0], data[:,1], bins=bins, range=(extenti, extentj))
    hitx = np.digitize(data[:, 0], edges[0])
    hity = np.digitize(data[:, 1], edges[1])
    hitbins = list(zip(hitx, hity))
    data_and_bins = list(zip(data, hitbins))
    it = range(1, bins+1)

    Combinations = []
    for bin in list(product(it,it)):
        test_item = []
        for item in data_and_bins:
            if item[1]==bin:
                test_item.append(item[0])
        choice = np.random.choice(list(range(len(test_item))))
        Combinations.append(test_item[choice])
    
    # where to save the data
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
        min_n_path = os.path.join(os.getcwd(), 'data/coupling-'+str(const['coupling'])+'/tmax-'+
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
        if not data_exists:
            print('-'*20+'Combination:{} out of {}, Initial (xA,xD):({:.3f},{:.3f})'.\
                format(i, len(Combinations)-1, ChiAInitial, ChiDInitial) + '-'*20)
            
        opt = Optimizer(ChiAInitial=ChiAInitial,
                        ChiDInitial=ChiDInitial,
                        DataExist=data_exists, 
                        data_path=data_path,
                        Case=i,
                        const=const,
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
            a = const['xA']
            d = const['xD']
            all_losses[i] = np.array([a, d, np.min(loss_data)])
        
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
