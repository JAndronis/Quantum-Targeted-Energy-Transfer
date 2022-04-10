import constants
from itertools import product
import numpy as np
import os
import gc
import time
import multiprocessing as mp
from Optimizer import mp_opt
from data_process import createDir

# Set globals
constants.setConstant('max_N', 3)
constants.setConstant('max_t', 25)
constants.setConstant('omegaA', 3)
constants.setConstant('omegaD', -3)
constants.setConstant('omegaMid', 0)
constants.setConstant('coupling', 0.01)
constants.setConstant('xMid', 0)
constants.setConstant('sites', 2)
constants.setConstant('resolution', 5)
CONST = constants.constants
constants.dumpConstants()

def getCombinations(a_lims, d_lims, method='bins', grid=2):
    """
    Creates a list of initial guess pairs to be fed to an optimizer call.

    Args:
        a_lims (list): list of 2 elements that contains the minimum 
                       and maximum xA's to use.
        d_lims (list): list of 2 elements that contains the minimum 
                       and maximum xD's to use.
        method (str, optional): Method to use for creating Combinations list. Defaults to 'bins'.
        grid (int, optional): Number of times to split the parameter space. Defaults to 2.

    Returns:
        list: A list of tuples, of all the initial guesses to try.
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
        xa = np.linspace(a_lims[0], a_lims[1], CONST['resolution'])
        xd = np.linspace(d_lims[0], d_lims[1], CONST['resolution'])
        Combinations = []
        for comb in product(xa, xd):
            Combinations.append(comb)

    return Combinations

if __name__=="__main__":

    # create data directory to save results
    data_path1 = os.path.join(os.getcwd(),'data')
    createDir(destination=data_path1, replace_query=False)

    # initialize helper parameters
    a_lims = [-10,10]   # limits of xA guesses
    d_lims = [-10,10]   # limits of xD guesses
    grid = 2
    edge = 1
    iteration = 0
    counter = 0
    done = False
    bin_choice = False
    grid_choice = False
    min_a, min_d, min_loss = 0, 0, CONST['max_N']   # initializing min_loss to the maximum number
                                                    # ensures that the initial combinations of initial
                                                    # guesses will be done with the bin method

    t0 = time.time()
    while not done and iteration<=5:

        # create directory of current iteration
        data_path2 = os.path.join(data_path1, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=False)

        # if min loss is not small enough do a general search with the bin method
        if grid<=6 and min_loss>=CONST['max_N']-1.5:
            if bin_choice and min_loss<=CONST['max_N']-0.1:
                # if this method has already been picked increase bin number
                if not grid==6:
                    edge /= iteration
                    grid += 2
                    a_min, a_max = min_a-edge, min_a+edge
                    d_min, d_max = min_d-edge, min_d+edge
                    a_lims = [a_min,a_max]
                    d_lims = [d_min,d_max]
                else:
                    grid = 2
                    edge = 1
                    iteration = 0
                    a_lims = [-10,10]
                    d_lims = [-10,10]
            Combinations = getCombinations(a_lims, d_lims, method='bins', grid=grid)
            iter = 1000
            bin_choice = True
            print(10*'-',f'Iteration: {iteration}, Method: Bins, Jobs: {len(Combinations)}', 10*'-')

        # else if min loss is suffeciently small do an exact search with the grid method
        elif min_loss<=CONST['max_N']-1.5:
            if grid_choice:
                # if this method has already been picked increase grid size
                CONST['resolution'] *= 2
                a_min, a_max = min_a-1, min_a+1
                d_min, d_max = min_d-1, min_d+1
                a_lims = [a_min,a_max]
                d_lims = [d_min,d_max]
            bin_choice = False
            Combinations = getCombinations(a_lims, d_lims, method='grid')
            iter = 200
            grid_choice = True
            print(10*'-',f'Iteration: {iteration}, Method: Grid, Jobs: {len(Combinations)}', 10*'-')

        t2 = time.time()
        try:
            # initialize processing pool
            pool = mp.Pool(mp.cpu_count()//2)

            # set input arg list for mp_opt() function
            args = [(i, ChiAInitial, ChiDInitial, data_path2, CONST, 'x1', 0.1, iter) for i, (ChiAInitial, ChiDInitial) in enumerate(Combinations)]

            # run multiprocess map 
            all_losses = pool.starmap_async(mp_opt, args).get()

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
    # end of main
    exit(0)
