import constants
from itertools import product
import numpy as np
import os
import time
import multiprocessing as mp
from Optimizer import mp_opt
from tet.data_process import createDir

if __name__=="__main__":
    ## enable memory growth
    #gpus = tf.config.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        # Currently, memory growth needs to be the same across GPUs
    #        for gpu in gpus:
    #            tf.config.experimental.set_memory_growth(gpu, True)
    #            logical_gpus = tf.config.list_logical_devices('GPU')
    #            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #    except RuntimeError as e:
    #        # Memory growth must be set before GPUs have been initialized
    #        print(e)
    
    # Set constants
    constants.setConstant('max_N', 3)
    constants.setConstant('max_t', 300)
    constants.setConstant('omegaA', 3)
    constants.setConstant('omegaD', -3)
    constants.setConstant('coupling', 0.1)
    constants.setConstant('xMid', 0)
    constants.setConstant('sites', 3)
    constants.setConstant('omegaMid', 3)
    constants.setConstant('resolution', 100)
    const = constants.constants
    constants.dumpConstants()
    done = False

    done = False
    edge = 1
    iteration = 0
    grid = 2

    data_path1 = os.path.join(os.getcwd(),'data')
    createDir(destination=data_path1)
    data_path2 = os.path.join(data_path1, f'try{iteration}')
    createDir(destination=data_path2)

    print('First set of tries...')

    # Make an initial search of the parameter space
    
    a_lims = [-10,10]
    d_lims = [-10,10]

    # make random initial guesses according to the number of bins
    xa = np.linspace(a_lims[0], a_lims[1], const['resolution'])
    xd = np.linspace(d_lims[0], d_lims[1], const['resolution'])
    data = np.array((list(product(xa,xd))))
    extenti = (a_lims[0]-0.1, a_lims[1]+0.1)
    extentj = (d_lims[0]-0.1, d_lims[1]+0.1)
    bins = grid
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

    # mp ---------------------------------------------------------------------------- #
    
    t0 = time.time()
    pool = mp.Pool(mp.cpu_count()//2)
    args = [(i, ChiAInitial, ChiDInitial, data_path2, const, 'x2', 0.1, 400) for i, (ChiAInitial, ChiDInitial) in enumerate(Combinations)]
    all_losses = pool.starmap_async(mp_opt, args).get()
    pool.close()
    t1 = time.time()
    dt = t1-t0

    # mp ---------------------------------------------------------------------------- #

    all_losses = np.array(all_losses)
    min_a, min_d = float(all_losses[np.argmin(all_losses[:,2]), 0]), float(all_losses[np.argmin(all_losses[:,2]), 1])
    min_loss = float(all_losses[np.argmin(all_losses[:,2]), 2])

    print(f"Best parameters of tries: loss={min_loss}, xA={min_a}, xD={min_d}")
    print("Code run time: ", dt, " s")

    iteration += 1
    if min_loss<=0.1:
        print('TET!')
        print(min_a, min_d, min_loss)
        done = True
        exit()
    else:
        while True:
            edge /= iteration
            if grid<6: grid += 2
            
            data_path2 = os.path.join(data_path1, f'try{iteration}')
            createDir(destination=data_path2)
            print('Try ', iteration)
            a_min, a_max = min_a-edge, min_a+edge
            d_min, d_max = min_d-edge, min_d+edge
            a_lims = [a_min,a_max]
            d_lims = [d_min,d_max]
            grid_size = grid

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

            # mp ---------------------------------------------------------------------------- #
            
            t0 = time.time()
            pool = mp.Pool(mp.cpu_count()//2)
            args = [(i, ChiAInitial, ChiDInitial, data_path2, const, 'x2', 0.1, 400) for i, (ChiAInitial, ChiDInitial) in enumerate(Combinations)]
            all_losses = pool.starmap_async(mp_opt, args).get()
            pool.close()
            t1 = time.time()
            dt = t1-t0

            # mp ---------------------------------------------------------------------------- #

            all_losses = np.array(all_losses)
            min_a, min_d = float(all_losses[np.argmin(all_losses[:,2]), 0]), float(all_losses[np.argmin(all_losses[:,2]), 1])
            min_loss = float(all_losses[np.argmin(all_losses[:,2]), 2])
            
            print(f"Best parameters of tries: loss={min_loss}, xA={min_a}, xD={min_d}")
            print("Code run time: ", dt, " s")

            iteration += 1
            
            if float(min_loss)<=0.1:
                print('TET!')
                print(min_a, min_d, min_loss)
                done = True
                break

            if iteration>=3:
                done = True
                break