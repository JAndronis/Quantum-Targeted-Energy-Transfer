from itertools import product
import numpy as np
import os

import constants
from Optimizer import Optimizer
from data_process import createDir, read_1D_data, PlotResults

def solver(a_lims, d_lims, iterations=500, learning_rate=0.01, create_plot=False):
    #* Load the parameters of the problem
    const = constants.loadConstants()
    done = False
    
    #* Make a grid of uniformly distributed initial parameter guesses
    xa = np.linspace(a_lims[0], a_lims[1], const['Npoints'])
    xd = np.linspace(d_lims[0], d_lims[1], const['Npoints'])
    Combinations = list(product(xa,xd))
    
    #* Init an array to save chiAs/chiDs and resulting losses
    all_losses = np.ones((len(Combinations), 3))*const['max_N']
    p = os.path.join(os.getcwd(), f'data')
    createDir(destination=p, replace_query=True)
        
    for i,(ChiAInitial, ChiDInitial) in enumerate(Combinations):
        if not done:
            #* Path for svaing the data for the ith iteration
            data_path = os.path.join(p, f'data_{i}')
            
            # # Read Data from the i-th combination
            # CombinationPath = os.path.join(data_path,f'combination_{i}')

            #* Check if the data file exists
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
            
            opt = Optimizer(target_site='x1',
                            DataExist=data_exists,
                            data_path=data_path,
                            const=const,
                            lr=learning_rate,
                            iterations=iterations)
            opt(ChiAInitial, ChiDInitial)
            
            # Load Data
            loss_data = read_1D_data(destination=data_path, name_of_file='losses.txt')
            a = const['xA']
            d = const['xD']
            all_losses[i] = np.array([a, d, np.min(loss_data)])
        
        #! If you want to create the heatmap when having the data
        if create_plot: 
            fig = PlotResults(const=const,path=data_path)
            fig.plot(ChiDInitial=ChiDInitial, ChiAInitial=ChiAInitial, xa_lims=[-5,5], xd_lims=[-5,5])
        
        # Finish iteration if TET is found
        if np.min(loss_data)<0.1:
            done=True
            break
    #print(np.argmin(all_losses[:,2])) 
    min_a, min_d = all_losses[np.argmin(all_losses[:,2]), 0], all_losses[np.argmin(all_losses[:,2]), 1]
    min_loss = all_losses[np.argmin(all_losses[:,2]), 2]
    return min_a, min_d, min_loss

if __name__=="__main__":
    CONST = constants.constants
    print(CONST)
    constants.dumpConstants()

    solver(a_lims=[-5,5], d_lims=[-5,5], create_plot=False)
    exit(0)
