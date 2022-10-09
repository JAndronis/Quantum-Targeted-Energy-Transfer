
import numpy as np
import os
import gc
import sys
import time
# import multiprocessing as mp
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from itertools import combinations, product
import tensorflow as tf

import constants
from Optimizer import mp_opt, Optimizer
from data_process import createDir, read_1D_data
from constants import solver_params,TensorflowParams

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#!Creates a list of initial guess pairs to be fed to an optimizer call
def getCombinations(TrainableVarsLimits, method='bins', grid=2):
    """
    Creates a list of initial guess pairs to be fed to an optimizer call

    Args:
        * TrainableVarsLimits (dict): The keys are the nonlinearity parameters of each site and the values 
        * include a list with the limits of the said variable.
        * const (dict): Dictionary of problem parameters.
        * method (str, optional): Method to use for creating Combinations list. Defaults to 'bins'.
        * grid (int, optional): Number of times to split the parameter space. Defaults to 2.

    Returns:
        * list: A list of tuples, of all the initial guesses to try.
    """
    
    method_list = solver_params['methods']

    if method not in method_list:
        raise ValueError('Provided method not in list of supported methods [\'grid\', \'bins\']')

    # Works only in the dimer case
    if method=='bins':
        TrainableSpans = [ np.linspace( TrainableVarsLimits[f'x{i}lims'][0], TrainableVarsLimits[f'x{i}lims'][1], solver_params['Npoints']) for i in TensorflowParams['train_sites'] ]
        data = np.array(list(product(*TrainableSpans)))
        data_list = [data[:, i] for i in range(data.shape[1])]
        
        # Extent of bins needs to be a bit smaller than parameter range
        extents = [ ( TrainableVarsLimits[f'x{i}lims'][0]-0.1, TrainableVarsLimits[f'x{i}lims'][1]+0.1 ) for i in TensorflowParams['train_sites'] ]
        
        # Produce bin edges.Default returning: H,xedges,yedges.
        _, *edges = np.histogramdd(data_list, bins=grid, range=extents)
        
        # create an indexed list of possible choices for initial guess
        hit = [np.digitize(data_list[i], edges[0][i]) for i,_ in enumerate(data_list)]
        hitbins = list(zip(*hit))
        data_and_bins = list(zip(data, hitbins))
        it = [ range(1, grid+1) for _ in range(len(TensorflowParams['train_sites'])) ]
        Combinations = []
        for bin in list(product(*it)):
            test_item = []
            for item in data_and_bins:
                if item[1]==bin:
                    test_item.append(item[0])
            # choose initial conditions and append them to the combination list
            choice = np.random.choice(list(range(len(test_item))))
            Combinations.append(test_item[choice])

    elif method=='grid':
        # make a grid of uniformly distributed initial parameter guesses
        TrainableSpans = [ np.linspace(TrainableVarsLimits[f'x{i}lims'][0], TrainableVarsLimits[f'x{i}lims'][1], solver_params['Npoints'])
            for i in TensorflowParams['train_sites'] ]

        Combinations = list(product(*TrainableSpans))

    return Combinations

def solver_mp(
    TrainableVarsLimits, const, grid=2, lr=TensorflowParams['lr'], method='bins',
    epochs_bins=solver_params['epochs_bins'], epochs_grid=solver_params['epochs_grid'], 
    target_site=0, main_opt=False, return_values=False, data_path=os.path.join(os.getcwd(),'data')
    ):

    """
    Function that utilizes multiple workers on the cpu to optimize the non linearity parameters for TET.
    It uses the Optimizer class to write trajectory data to multiple files so it can be parsed later if needed.

    Args:
        * TrainableVarsLimits (Dictionary): The keys are the nonlinearity parameters of each site and the values 
        * include a list with the limits of the said variable.
        * const (dict): Dictionary of system parameters that follows the convention used by the tet.constants() module.
        * grid (int, optional): Integer representing the number of times to split the parameter space. Defaults to 2.
        * lr (float, optional): Learning rate of the optimizer. Defaults to 0.1.
        * epochs_bins (int, optional): Epochs that the optimizer is going to run for using the bins method for initial guesses. Defaults to 1000.
        * epochs_grid (int, optional): Epochs that the optimizer is going to run for using the grid method for initial guesses. Defaults to 200.
        * target_site (str, optional): Target site for the optimizer to monitor. Defaults to 'x0' aka the 'donor' site.
        * main_opt (bool, optional): If to further optimize with an optimizer with initial guesses provided by the best performing test optimizer.
        * return_values(bool, optional): If to return the modified constants dictionary.
        * data_path (str, optional): Path to create the data directory. Defaults to cwd/data.
    """

    #! Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create data directory to save results
    createDir(destination=data_path, replace_query=True)

    # Initialize helper parameters
    lims = list(TrainableVarsLimits.values())

    # _edge =  [0.1,0.5,1.5,2.5,3.0,3.5,4.0]  
    _edge =  [4.0,3.5,3.0,2.5,1.5,0.5,0.1]

    # Control how many times loss is lower than the threshold having changed the limits
    iteration = 0

    # Count attempts based on the limits 
    counter = 0
    done = False
    bin_choice = False

    #* An array to save the optimal parameters
    OptimalVars, min_loss = np.zeros(len(TrainableVarsLimits)), const['max_N']  # initializing min_loss to the maximum number
                                                                                # ensures that the initial combinations of initial
                                                                                # guesses will be done with the bin method

    t0 = time.time()
    while not done and iteration < 1:

        # Create directory of current iteration
        data_path2 = os.path.join(data_path, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=True)
        
        Combinations = getCombinations(TrainableVarsLimits, method=method, grid=grid)
        if method=='bins': iterations = epochs_bins
        else: iterations = epochs_grid
        #grid_choice = True
        print(10*'-',f'Iteration: {iteration}, Method: {method}, Jobs: {len(Combinations)}, lims: {lims}', 10*'-')

        t2 = time.time()
        # Initialize processing pool
        with MPIPoolExecutor(max_workers=MPI.COMM_WORLD.Get_size(), root=0) as executor:
            # Set input arg list for mp_opt() function
            args = [(i, combination, data_path2, const, target_site, iterations) for i, (combination) in enumerate(Combinations)]

            # Run multiprocess map
            _all_losses = executor.starmap(mp_opt, args)

        t3 = time.time()

        # Collect code run time for optimization
        dt = t3-t2  

        # Gather results
        all_losses = np.fromiter(_all_losses, float)
        OptimalVars = [float(all_losses[np.argmin(all_losses[:,const['sites']]), i]) for i in range(const['sites'])]
        min_loss = float(all_losses[np.argmin(all_losses[:,const['sites']]), const['sites']])
        
        # Print results of run
        print(f"Best parameters of tries: loss={min_loss}, OptimalVars = {OptimalVars}")
        print("Code run time: ", dt, " s")

        if min_loss<=const["max_N"]/2:
            counter += 1
            edge = _edge[iteration]
            #grid += 2
            lims = [[OptimalVars[i]-edge, OptimalVars[i]+edge] for i in range(len(TrainableVarsLimits))]

        else:
            solver_params['Npoints'] += 1
            lr += 0.1

        # advance iteration
        iteration += 1

        # if loss has not been reduced for more than 5 iterations stop
        if counter>=5 and min_loss>=CONST['max_N']/2:
            print("Couldn't find TET")
            break
        
        # tet has been achieved no need to continue
        if float(min_loss)<=0.1:
            print('TET!')
            print(f'OptimalParams:{OptimalVars}')
            done = True
            break

    t1 = time.time()

    const['chis'] = OptimalVars
    const['min_n'] = min_loss

    if main_opt:
        data_path3 = os.path.join(data_path, f'main_opt')
        _opt = Optimizer(
            target_site=solver_params['target'], DataExist=False, 
            const=const, Print=True, data_path=data_path3
        )
        _opt(*OptimalVars)

        #! Load Data
        loss_data = read_1D_data(
            destination=data_path3, name_of_file='losses.txt'
        )

        OptimalVars = read_1D_data(
            destination=data_path3, name_of_file='optimalvars.txt'
        )

        const['chis'] = OptimalVars
        const['min_n'] = min(loss_data)
    
    constants.dumpConstants(dict=const, path=data_path)
    
    print('Total solver run time: ', t1-t0)

    if return_values:
        return const.copy()

if __name__=="__main__":

    import constants
    import matplotlib.pyplot as plt
    #! Import the constants of the problem
    CONST = constants.constants

    for CONST['max_N'] in range(1, 2):
        for CONST['omegas'][0] in range(1, 2):
            CONST['omegas'][-1] = -CONST['omegas'][0]
            CONST['omegas'][1] = CONST['omegas'][-1]
            xd = (CONST['omegas'][-1] - CONST['omegas'][0])/CONST['max_N']
            xa = -xd
            CONST['chis'] = [xd, 0, xa]

            # create data directory with the naming convention data_{unix time}
            data_dir_name = f'data_{time.time_ns()}'
            data = os.path.join(os.getcwd(), data_dir_name)

            #! Call the solver function that uses multiprocessing(pointer _mp)
            solver_mp(
                {'x1lims': [-40, 40]}, const=CONST, 
                target_site=solver_params['target'], data_path=data,
                grid=4, lr=0.6
            )
