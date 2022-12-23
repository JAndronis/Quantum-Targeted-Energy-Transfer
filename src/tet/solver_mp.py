from typing import Any
import numpy as np
import os
import gc
import time
import multiprocessing as mp
from itertools import product
import tensorflow as tf

from .Optimizer import mp_opt, Optimizer
from .data_process import createDir, read_1D_data
from .constants import solver_params, TensorflowParams, dumpConstants

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def getCombinations(
        trainable_vars_limits, train_sites=TensorflowParams['train_sites'],
        method='bins', grid=solver_params['Npoints']
):
    """
    Creates a list of initial guess pairs to be fed to an optimizer call

    Args:
        train_sites (list): Sites of nonlinearity parameters to train.
        trainable_vars_limits (dict): The keys are the nonlinearity parameters of each site and the values include
        a list with the limits of the said variable.
        method (str, optional): Method to use for creating Combinations list. Defaults to 'bins'.
        grid (int, optional): Number of times to split the parameter space. Defaults to 2.

    Returns:
        list: A list of tuples, of all the initial guesses to try.
    """

    method_list = solver_params['methods']

    if method not in method_list:
        raise ValueError('Provided method not in list of supported methods [\'grid\', \'bins\']')

    # Works only in the dimer case
    if method == 'bins':
        trainable_spans = [
            np.linspace(trainable_vars_limits[f'x{i}lims'][0], trainable_vars_limits[f'x{i}lims'][1], grid)
            for i in train_sites
        ]
        data = np.array(list(product(*trainable_spans)))
        data_list = [data[:, i] for i in range(data.shape[1])]

        # Extent of bins needs to be a bit smaller than parameter range
        extents = [
            (trainable_vars_limits[f'x{i}lims'][0] - 0.1, trainable_vars_limits[f'x{i}lims'][1] + 0.1)
            for i in train_sites
        ]

        # Produce _bin edges.Default returning: H,xedges,yedges.
        _, *edges = np.histogramdd(data_list, bins=grid, range=extents)

        # create an indexed list of possible choices for initial guess
        hit = [np.digitize(data_list[i], edges[0][i]) for i, _ in enumerate(data_list)]
        hitbins = list(zip(*hit))
        data_and_bins = list(zip(data, hitbins))
        it = [range(1, grid + 1) for _ in train_sites]
        combinations = []
        for _bin in list(product(*it)):
            test_item = []
            for item in data_and_bins:
                if item[1] == _bin:
                    test_item.append(item[0])
            # choose initial conditions and append them to the combination list
            if len(test_item) != 0:
                choice = np.random.choice(list(range(len(test_item))))
                combinations.append(test_item[choice])

        return combinations

    elif method == 'grid':
        # make a grid of uniformly distributed initial parameter guesses
        trainable_spans = [
            np.linspace(trainable_vars_limits[f'x{i}lims'][0], trainable_vars_limits[f'x{i}lims'][1], grid)
            for i in train_sites
        ]

        combinations = list(product(*trainable_spans))

        return combinations


def solver_mp(
        trainable_vars_limits: dict, const: dict, grid=2, lr=0.1, beta_1=0.9, amsgrad=False,
        write_data=False, iterations=1, method='bins', epochs_bins=solver_params['epochs_bins'],
        epochs_grid=solver_params['epochs_grid'], target_site=0, main_opt=False,
        return_values=False, data_path=os.path.join(os.getcwd(), 'data'), cpu_count=mp.cpu_count() // 2
) -> dict[float, Any]:
    """
    Function that utilizes multiple workers on the cpu to optimize the nonlinearity parameters for TET.
    It uses the Optimizer class to write trajectory data to multiple files, so it can be parsed later if needed.

    Args:
        trainable_vars_limits (Dictionary): The keys are the nonlinearity parameters of each site and the values include
        a list with the limits of the said variable.
        const (dict): Dictionary of system parameters that follows the convention used by the tet.constants() module.
        grid (int, optional): Integer representing the number of times to split the parameter space. Defaults to 2.
        lr (float, optional): Learning rate of the optimizer. Defaults to 0.1.
        beta_1 (float, optional): beta_1 parameter of ADAM. Defaults to 0.9.
        amsgrad (bool, optional): Whether to use the amsgrad version of ADAM. Defaults to False.
        write_data (bool, optional): Whether to write trajectory and loss data of the optimizers in files. Defaults to
        False.
        iterations (int, optional): Number of parallel iterations of optimizers. Defaults to 1.
        method (str, optional): Defines the method of optimization to be used. Defaults to 'bins'.
        epochs_bins (int, optional): Epochs that the optimizer is going to run for using the bins method
        for initial guesses. Defaults to 1000.
        epochs_grid (int, optional): Epochs that the optimizer is going to run for using the grid method
        for initial guesses. Defaults to 200.
        target_site (str, optional): Target site for the optimizer to monitor. Defaults to 'x0' aka the 'donor' site.
        main_opt (bool, optional): If to further optimize with an optimizer with initial guesses provided by the best
        performing test optimizer.
        return_values(bool, optional): If to return the modified constants dictionary.
        data_path (str, optional): Path to create the data directory. Defaults to cwd/data.
        cpu_count (int, optional): Number of cpu cores to use for multiprocessing
    
    Returns:
        dict: If return_values is True, return a dictionary of the resulting parameters of the optimization process.
    """

    # ! Use cpu since we are doing parallelization on the cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create data directory to save results
    createDir(destination=data_path, replace_query=False)

    # Initialize helper parameters
    lims = list(trainable_vars_limits.values())

    # _edge =  [0.1,0.5,1.5,2.5,3.0,3.5,4.0]  
    _edge = [4.0, 3.5, 3.0, 2.5, 1.5, 0.5, 0.1]

    # Control how many times loss is lower than the threshold having changed the limits
    iteration = 0

    # Count attempts based on the limits 
    lim_changes = 0

    # An array to save the optimal parameters
    optimal_vars, min_loss = np.zeros(len(trainable_vars_limits)), const['max_N']
    # initializing min_loss to the maximum number
    # ensures that the initial combinations of initial
    # guesses will be done with the bin method

    # get train_sites from limits of trainable parameters by parsing elements in strings
    train_sites = [int(list(trainable_vars_limits.keys())[i][1]) for i in range(len(trainable_vars_limits))]

    t0 = time.time()
    while iteration < iterations:

        # Create directory of current iteration
        data_path2 = os.path.join(data_path, f'iteration_{iteration}')
        createDir(destination=data_path2, replace_query=False)

        combinations = getCombinations(trainable_vars_limits, train_sites=train_sites, method=method, grid=grid)
        if method == 'bins':
            epochs = epochs_bins
        else:
            epochs = epochs_grid
        # grid_choice = True
        print(10 * '-', f'Iteration: {iteration}, Method: {method}, Jobs: {len(combinations)}, lims: {lims}', 10 * '-')

        t2 = time.time()
        # Initialize processing pool
        pool = mp.Pool(cpu_count)
        # pool = mp.Pool(mp.cpu_count())

        # Set input arg list for mp_opt() function
        args = [
            (i, combination, data_path2, const, target_site,
             epochs, lr, beta_1, amsgrad, write_data, train_sites)
            for i, (combination) in enumerate(combinations)
        ]

        try:
            # Run multiprocess map 
            _all_losses = pool.starmap_async(mp_opt, args).get()
        finally:
            # Make sure to close pool so no more processes start
            pool.close()
            pool.join()
            # Garbage collector
            gc.collect()

        t3 = time.time()

        # Collect code run time for optimization
        dt = t3 - t2

        # Gather results
        all_losses = np.array(_all_losses)
        optimal_vars = [float(all_losses[np.argmin(all_losses[:, const['sites']]), i]) for i in range(const['sites'])]
        min_loss = float(all_losses[np.argmin(all_losses[:, const['sites']]), const['sites']])

        # Print results of run
        print(f"Best parameters of tries: loss={min_loss}, optimal_vars = {optimal_vars}")
        print("Code run time: ", dt, " s")

        if min_loss <= const["max_N"] / 2:
            lim_changes += 1
            edge = _edge[iteration]
            # grid += 2
            lims = [[optimal_vars[i] - edge, optimal_vars[i] + edge] for i in range(len(trainable_vars_limits))]

        else:
            lr += 0.1

        # advance iteration
        iteration += 1

        # if loss has not been reduced for more than 5 iterations stop
        if lim_changes >= 5 and min_loss >= const['max_N'] / 2:
            print("Couldn't find TET")
            break

        # tet has been achieved no need to continue
        if float(min_loss) <= 0.1:
            print('TET!')
            print(f'OptimalParams:{optimal_vars}')
            break

    t1 = time.time()

    const['chis'] = optimal_vars
    const['min_n'] = min_loss

    if main_opt:
        data_path3 = os.path.join(data_path, f'main_opt')
        _opt = Optimizer(
            target_site=solver_params['target'], DataExist=False,
            const=const, Print=True, data_path=data_path3
        )
        _opt(*optimal_vars)

        # ! Load Data
        loss_data = read_1D_data(
            destination=data_path3, name_of_file='losses.txt'
        )

        optimal_vars = read_1D_data(
            destination=data_path3, name_of_file='optimalvars.txt'
        )

        const['chis'] = optimal_vars
        const['min_n'] = min(loss_data)

    dumpConstants(dict_name=const, path=data_path)

    print('Total solver run time: ', t1 - t0)

    if return_values:
        return const
