
import numpy as np
import os
import gc
import sys
import time
import multiprocessing as mp
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
        TrainableSpans = [ np.linspace( TrainableVarsLimits[f'x{i}lims'][0], TrainableVarsLimits[f'x{i}lims'][1], solver_params['Npoints'])
                            for i in TensorflowParams['train_sites'] ]

        Combinations = list(product(*TrainableSpans))

    return Combinations

def solver_mp(TrainableVarsLimits, const, 
              grid=2, lr=TensorflowParams['lr'],
              epochs_bins=solver_params['epochs_bins'], epochs_grid=solver_params['epochs_grid'], 
              target_site='x0', 
              data_path=os.path.join(os.getcwd(),'data')):

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
        * data_path (str, optional): Path to create the data directory. Defaults to cwd/data.
    """

    #! Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create data directory to save results
    createDir(destination=data_path, replace_query=True)

    # Initialize helper parameters
    lims = list(TrainableVarsLimits.values())
    grid = grid

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
    while not done and iteration < 2:

        # Create directory of current iteration
        data_path2 = os.path.join(data_path, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=True)
        """
        # if min loss is not small enough do a general search with the bin method
        if grid<=6 and min_loss>=const['max_N']-1.5:
            # if bin_choice:
            #     # if this method has already been picked increase bin number
            #     if not grid==6:
            #         edge = _edge[iteration]
            #         grid += 2
            #         a_min, a_max = min_a-edge, min_a+edge
            #         d_min, d_max = min_d-edge, min_d+edge
            #         a_lims = [a_min,a_max]
            #         d_lims = [d_min,d_max]
            #     else:
            #         grid = 2
            #         iteration = 0
            #         a_lims = xa_lims
            #         d_lims = xd_lims
            Combinations = getCombinations(TrainableVarsLimits, method='bins', grid=grid)
            # iter = epochs_bins
            # bin_choice = True
            # print(10*'-',f'Iteration: {iteration}, Method: Bins({grid*2}), Jobs: {len(Combinations)}, a_lim: {a_lims}, d_lim: {d_lims}', 10*'-')

        # # else if min loss is suffeciently small do an exact search with the grid method
        # elif min_loss<=const['max_N']-1.5:
            # if grid_choice:
            #     # if this method has already been picked increase grid size
            #     const['Npoints'] *= 2
            #     a_min, a_max = min_a-1, min_a+1
            #     d_min, d_max = min_d-1, min_d+1
            #     a_lims = [a_min,a_max]
            #     d_lims = [d_min,d_max]
        #bin_choice = False
        """
        
        Combinations = getCombinations(TrainableVarsLimits, method='grid')
    
        iter = epochs_grid
        #grid_choice = True
        print(10*'-',f'Iteration: {iteration}, Method: Grid, Jobs: {len(Combinations)}, lims: {lims}', 10*'-')

        t2 = time.time()
        # Initialize processing pool
        pool = mp.Pool(max(mp.cpu_count()//4, 1))
        # pool = mp.Pool(mp.cpu_count())

        # Set input arg list for mp_opt() function
        args = [(i, combination, data_path2, const, target_site, lr, iter) for i, (combination) in enumerate(Combinations)]

        try:
            # Run multiprocess map 
            _all_losses = pool.starmap_async(mp_opt, args).get()

        except KeyboardInterrupt():
            print('Keyboard Interrupt.')
            # Make sure to close pool so no more processes start
            pool.close()    
            pool.join()
            gc.collect()
            sys.exit(1)

        finally:
            # Make sure to close pool so no more processes start
            pool.close()    
            pool.join()
            # Garbage collector
            gc.collect()    

        t3 = time.time()
        # Collect code run time for optimization
        dt = t3-t2  

        # Gather results
        all_losses = np.array(_all_losses)
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

    data_path3 = os.path.join(data_path, f'main_opt')
    _opt = Optimizer(target_site=solver_params['target'], DataExist=False, const=const, Print=True, data_path=data_path3)
    _opt(*OptimalVars)

    #! Load Data
    loss_data = read_1D_data(destination=data_path3, name_of_file='losses.txt')
    OptimalVars = read_1D_data(destination=data_path3, name_of_file='optimalvars.txt')

    const['chis'] = OptimalVars
    const['min_n'] = min(loss_data)
    constants.dumpConstants(dict=const, path=data_path)
    
    print('Total solver run time: ', t1-t0)


if __name__=="__main__":

    #! Import the constants of the problem
    CONST = constants.constants

    # create data directory with the naming convention data_{unix time}
    data_dir_name = f'data_{time.time_ns()}'
    data = os.path.join(os.getcwd(), data_dir_name)

    #! Call the solver function that uses multiprocessing(pointer _mp)
    solver_mp(constants.TrainableVarsLimits, const=CONST, target_site=solver_params['target'], data_path=data)

    exit(0)
