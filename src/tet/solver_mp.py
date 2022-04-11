from itertools import product
import numpy as np
import os
import gc
import sys
import time
import multiprocessing as mp

from tet.Optimizer import mp_opt
from tet.data_process import createDir
from tet import constants

def getCombinations(a_lims, d_lims, method='bins', grid=2, const=constants.loadConstants()):
    """
    Creates a list of initial guess pairs to be fed to an optimizer call.

    Args:
        * a_lims (list): list of 2 elements that contains the minimum 
                       and maximum xA's to use.
        * d_lims (list): list of 2 elements that contains the minimum 
                       and maximum xD's to use.
        * method (str, optional): Method to use for creating Combinations list. Defaults to 'bins'.
        grid (int, optional): Number of times to split the parameter space. Defaults to 2.

    Returns:
        * list: A list of tuples, of all the initial guesses to try.
    """
    
    if method=='bins':
        xa = np.linspace(a_lims[0], a_lims[1], 100)
        xd = np.linspace(d_lims[0], d_lims[1], 100)
        data = np.array(list(product(xa,xd)))
        
        # extent of bins needs to be a bit smaller than parameter range
        extenti = (a_lims[0]-0.1, a_lims[1]+0.1)
        extentj = (d_lims[0]-0.1, d_lims[1]+0.1)
        
        # produce bin edges
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

    elif method=='grid':
        # make a grid of uniformly distributed initial parameter guesses
        xa = np.linspace(a_lims[0], a_lims[1], const['resolution'])
        xd = np.linspace(d_lims[0], d_lims[1], const['resolution'])
        Combinations = []
        for comb in product(xa, xd):
            Combinations.append(comb)

    return Combinations

def solver_mp(xa_lims, xd_lims, const=constants.loadConstants(), grid=2, lr=0.1,\
     epochs_bins=1000, epochs_grid=200, target_site='x0', data_path=os.path.join(os.getcwd(),'data')):
    """
    Function that utilizes multiple workers on the cpu to optimize the non linearity parameters for TET.
    It uses the Optimizer class to write trajectory data to multiple files so it can be parsed later if needed.

    Args:
        * xa_lims (list): List of 2 elements. Contains the limits of the xA parameter space to search in.
        * xd_lims (list): List of 2 elements. Contains the limits of the xD parameter space to search in.
        * const (dict): Dictionary of system parameters that follows the convention used by the
                        tet.constants() module. Defaults to constants.loadConstants().
        * grid (int, optional): Integer representing the number of times to split the parameter space. Defaults to 2.
        * lr (float, optional): Learning rate of the optimizer. Defaults to 0.1.
        * epochs_bins (int, optional): Epochs that the optimizer is going to run for using the bins method for initial guesses. Defaults to 1000.
        * epochs_grid (int, optional): Epochs that the optimizer is going to run for using the grid method for initial guesses. Defaults to 200.
        * target_site (str, optional): Target site for the optimizer to monitor. Defaults to 'x0' aka the 'donor' site.
        * data_path (str, optional): Path to create the data directory. Defaults to cwd/data.
    """

    # use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # create data directory to save results
    data_path1 = data_path
    createDir(destination=data_path1, replace_query=True)

    # initialize helper parameters
    a_lims = xa_lims   # limits of xA guesses
    d_lims = xd_lims   # limits of xD guesses
    grid = grid
    _edge = [5, 4, 3, 2, 1, 0.5, 0.1]   # list of integers to decrease bin size by
    iteration = 0
    counter = 0
    done = False
    bin_choice = False
    grid_choice = False
    min_a, min_d, min_loss = 0, 0, const['max_N']   # initializing min_loss to the maximum number
                                                    # ensures that the initial combinations of initial
                                                    # guesses will be done with the bin method

    t0 = time.time()
    while not done and iteration<=5:

        # create directory of current iteration
        data_path2 = os.path.join(data_path1, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=True)

        # if min loss is not small enough do a general search with the bin method
        if grid<=6 and min_loss>=const['max_N']-1.5:
            if bin_choice and min_loss<=const['max_N']:
                # if this method has already been picked increase bin number
                if not grid==6:
                    edge = _edge[iteration]
                    grid += 2
                    a_min, a_max = min_a-edge, min_a+edge
                    d_min, d_max = min_d-edge, min_d+edge
                    a_lims = [a_min,a_max]
                    d_lims = [d_min,d_max]
                else:
                    grid = 2
                    edge = 1
                    iteration = 0
                    a_lims = xa_lims
                    d_lims = xd_lims
            Combinations = getCombinations(a_lims, d_lims, method='bins', grid=grid)
            iter = epochs_bins
            bin_choice = True
            print(10*'-',f'Iteration: {iteration}, Method: Bins({grid*2}), Jobs: {len(Combinations)}, a_lim: {a_lims}, d_lim: {d_lims}', 10*'-')

        # else if min loss is suffeciently small do an exact search with the grid method
        elif min_loss<=const['max_N']-1.5:
            if grid_choice:
                # if this method has already been picked increase grid size
                const['resolution'] *= 2
                a_min, a_max = min_a-1, min_a+1
                d_min, d_max = min_d-1, min_d+1
                a_lims = [a_min,a_max]
                d_lims = [d_min,d_max]
            bin_choice = False
            Combinations = getCombinations(a_lims, d_lims, method='grid')
            iter = epochs_grid
            grid_choice = True
            print(10*'-',f'Iteration: {iteration}, Method: Grid, Jobs: {len(Combinations)}, a_lim: {a_lims}, d_lim: {d_lims}', 10*'-')

        t2 = time.time()
        try:
            # initialize processing pool
            pool = mp.Pool(mp.cpu_count()//2)

            # set input arg list for mp_opt() function
            args = [(i, ChiAInitial, ChiDInitial, data_path2, const, target_site, lr, iter) for i, (ChiAInitial, ChiDInitial) in enumerate(Combinations)]

            # run multiprocess map 
            all_losses = pool.starmap_async(mp_opt, args).get(timeout=500)

        except KeyboardInterrupt:
            print('Keyboard Interrupt.')
            sys.exit(1)

        finally:
            pool.close()    # make sure to close pool so no more processes start
            pool.join()
            gc.collect()    # garbage collector

        t3 = time.time()
        dt = t3-t2  # collect code run time for optimization

        # gather results
        all_losses = np.array(all_losses)
        min_a, min_d = float(all_losses[np.argmin(all_losses[:,2]), 0]), float(all_losses[np.argmin(all_losses[:,2]), 1])
        min_loss = float(all_losses[np.argmin(all_losses[:,2]), 2])
        
        # print results of run
        print(f"Best parameters of tries: loss={min_loss}, xA={min_a}, xD={min_d}")
        print("Code run time: ", dt, " s")

        # advance iteration
        iteration += 1
        counter += 1

        # if loss has not been reduced for more than 5 iterations stop
        if counter>=5 and min_loss>=CONST['max_N']/2:
            print('Couldnt find TET')
            break
        
        # tet has been achieved no need to continue
        if float(min_loss)<=0.1:
            print('TET!')
            print(f"xA: {min_a}, xD: {min_d}, loss: {min_loss}")
            done = True
            break

    t1 = time.time()
    print('Total solver run time: ', t1-t0)
