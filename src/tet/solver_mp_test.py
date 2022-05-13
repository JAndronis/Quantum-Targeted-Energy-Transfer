from itertools import combinations, product
import numpy as np
import os

import gc
import sys
import time
import multiprocessing as mp


from Optimizer import mp_opt
from data_process import createDir
import constants
from constants import solver_params,TensorflowParams


#!Creates a list of initial guess pairs to be fed to an optimizer call
def getCombinations(TrainableVarsLimits, const, method='bins', grid=2):
    """
    Args:
        * TrainableVarsLimits (Dictionary): The keys are the nonlinearity parameters of each site and the values 
        * include a list with the limits of the said variable.
        * const (dict): Dictionary of problem parameters.
        * method (str, optional): Method to use for creating Combinations list. Defaults to 'bins'.
        grid (int, optional): Number of times to split the parameter space. Defaults to 2.

    Returns:
        * list: A list of tuples, of all the initial guesses to try.
    """
    
    method_list = solver_params['methods']

    if method not in method_list:
        raise ValueError('Provided method not in list of supported methods [\'grid\', \'bins\']')


    if method=='bins':
        pass
        """
            xa = np.linspace(a_lims[0], a_lims[1], 100)
            xd = np.linspace(d_lims[0], d_lims[1], 100)
            data = np.array(list(product(xa,xd)))
            
            # Extent of bins needs to be a bit smaller than parameter range
            extenti = (a_lims[0]-0.1, a_lims[1]+0.1)
            extentj = (d_lims[0]-0.1, d_lims[1]+0.1)
            
            # Produce bin edges.Default returning: H,xedges,yedges.
            _, *edges = np.histogram2d(data[:,0], data[:,1], bins=grid, range=(extenti, extentj))
            
            # create an indexed list of possible choices for initial guess
            hitx = np.digitize(data[:, 0], edges[0])
            hity = np.digitize(data[:, 1], edges[1])
            hitbins = list(zip(hitx, hity))
            data_and_bins = list(zip(data, hitbins))
            it = range(1, grid+1)
            Combinations = []
            for bin in list(product(it,it)):
                test_item = []
                for item in data_and_bins:
                    if item[1]==bin:
                        test_item.append(item[0])
                # choose initial conditions and append them to the combination list
                choice = np.random.choice(list(range(len(test_item))))
                Combinations.append(test_item[choice])
            """

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


    #* use cpu since we are doing parallelization on the cpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create data directory to save results
    createDir(destination=data_path, replace_query=True)

    # Initialize helper parameters
    lims = list(TrainableVarsLimits.values())
    grid = grid

    _edge = [5, 4, 3, 2, 1, 0.5, 0.1]   # list of integers to decrease bin size by

    iteration,counter = 0,0
    done = False
    bin_choice = False

    #* An array to save the optimal parameters
    OptimalVars, min_loss = np.zeros(len(TrainableVarsLimits)), const['max_N']   # initializing min_loss to the maximum number
                                                    # ensures that the initial combinations of initial
                                                    # guesses will be done with the bin method

    t0 = time.time()
    while not done and iteration < 2:

        # Create directory of current iteration
        data_path2 = os.path.join(data_path, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=True)
        """
        # if min loss is not small enough do a general search with the bin method
        # if grid<=6 and min_loss>=const['max_N']-1.5:
        #     if bin_choice:
        #         # if this method has already been picked increase bin number
        #         if not grid==6:
        #             edge = _edge[iteration]
        #             grid += 2
        #             a_min, a_max = min_a-edge, min_a+edge
        #             d_min, d_max = min_d-edge, min_d+edge
        #             a_lims = [a_min,a_max]
        #             d_lims = [d_min,d_max]
        #         else:
        #             grid = 2
        #             iteration = 0
        #             a_lims = xa_lims
        #             d_lims = xd_lims
        #     Combinations = getCombinations(a_lims, d_lims, method='bins', grid=grid, const=const)
        #     iter = epochs_bins
        #     bin_choice = True
        #     print(10*'-',f'Iteration: {iteration}, Method: Bins({grid*2}), Jobs: {len(Combinations)}, a_lim: {a_lims}, d_lim: {d_lims}', 10*'-')

        # else if min loss is suffeciently small do an exact search with the grid method
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
        Combinations = getCombinations(TrainableVarsLimits, method='grid', const=const)
    
        iter = epochs_grid
        grid_choice = True
        print(10*'-',f'Iteration: {iteration}, Method: Grid, Jobs: {len(Combinations)}, lims: {lims}', 10*'-')

        t2 = time.time()
        # Initialize processing pool
        pool = mp.Pool(max(mp.cpu_count()//2, 1))

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
        OptimalVars = [float(all_losses[np.argmin(all_losses[:,const['sites']]), i]) for i in range(const['sites']) ]
        min_loss = float(all_losses[np.argmin(all_losses[:,const['sites']]), const['sites']])
        
        # print results of run
        print(f"Best parameters of tries: loss={min_loss}, OptimalVars = {OptimalVars}")
        print("Code run time: ", dt, " s")

        if min_loss<=const["max_N"]/2:
            counter += 1
            edge = _edge[iteration]
            #grid += 2
    
            lims = [ [ OptimalVars[i]-edge, OptimalVars[i]+edge ] for i in range(len(TrainableVarsLimits)) ]

        else:
            solver_params['Npoints'] += 1
            lr += 0.1

        # advance iteration
        iteration += 1

        # if loss has not been reduced for more than 5 iterations stop
        if counter>=5 and min_loss>=CONST['max_N']/2:
            print('Couldnt find TET')
            break
        
        # tet has been achieved no need to continue
        if float(min_loss)<=0.1:
            print('TET!')
            #print(f"xA: {min_a}, xD: {min_d}, loss: {min_loss}")
    
            print(f'OptimalParams:{OptimalVars}')
            done = True
            break

    t1 = time.time()

    # Write best parameters to parameter json
    const['chis'] = [f'{OptimalVars[i]}' for i in range(len(OptimalVars))]
    constants.dumpConstants(dict=const, path=data_path)

    print('Total solver run time: ', t1-t0)


if __name__=="__main__":
    #! Import the constants of the problem
    CONST = constants.constants

    #! Create a dictionary with the limits of each variable explored
    keys = [ f'x{i}lims' for i in TensorflowParams['train_sites'] ] 
    lims = [[-5,5]]*len(keys)
    TrainableVarsLimits = dict(zip(keys,lims))

    # create data directory with the naming convention data_{unix time}
    data_dir_name = f'data_{time.time_ns()}'
    data = os.path.join(os.getcwd(), data_dir_name)

    #! Call the solver function that uses multiprocessing(pointer _mp)
    solver_mp(TrainableVarsLimits, const=CONST, target_site=solver_params['target'], data_path=data)

    exit(0)
