import os
import json
from tensorflow import float32

# -------------------------------------------------------------------#

"""
Constants: A dictionary that defined the constant parameters of the problem
Documentation:
    * max_N: The number of bosons belonging to the system
    * max_t: Period of time to compute the loss function
    * omegas: An f-dimensional list with the frequency of the oscillators of the system
    * chis: An f-dimensional with the non linearity parameters of the oscillators. Include the constant value of 
    * the non trainable parameters and enter random values for the trainable ones.
    * sites: The number of the oscillators of the system. Usually denoted by f.
"""
system_constants = {'max_N': 3,
             'max_t': 25, 
             'omegas': [-3,3,3],
             'chis': [1.5,0,-1.5],
             'coupling': 1,
             'sites': 3,
             'timesteps': 30}

# -------------------------------------------------------------------#

"""
TensorflowParams: A dictionary that includes the parameters used for the commands of the Tensorflow library
Documentation:
    * DTYPE: dtype of parameters
    * lr: The default value of the learning rate of each optimizer.
    * iterations: The default number of maximum Iterations of each optimizer
    * tol : The tolerance of each optimizer concerning the changes in the non-linearity parameters.
    * train_sites: A list including the non-linearity parameters to be optimized. Begin counting from 0
"""
TensorflowParams = {'DTYPE': float32, 
                    'lr': 0.1, 
                    'iterations': 1000,
                    'tol':1e-8,
                    'train_sites': [1]}

# Define the acceptor and the donor site
acceptor = 'x{}'.format(system_constants['sites']-1)
donor = 'x0'

# -------------------------------------------------------------------#

"""
solver_params: A dictionary that defines a set of parameters used in the solver_mp.py file.
Documentation:
    * methods: Must remain immutable. A list with the possible methods of setting the initial guesses of the optimizers.
    * target: An integer ranging from 0 to f-1. It determines the site of which you desire to compute the loss function.
    * Default value is f-1,meaning the acceptor. 
    * Npoints: The num parameter in np.linspace. Given the limits of a trainable parameter, it determines in how many points
    * will you split the interval
    * epochs_grid: The maximum iterations of each optimizer when using the grid method
    * epochs_bins: The maximum iterations of each optimizer when using the bins method

"""
solver_params = {'methods': ['grid', 'bins'],
                'target': acceptor,
                'Npoints': 4,
                'epochs_grid':500,
                'epochs_bins':1000}

#! Create a dictionary with the limits of each variable explored
keys = [f'x{i}lims' for i in TensorflowParams['train_sites']] 
lims = [[-10,10]]*len(keys)
TrainableVarsLimits = dict(zip(keys,lims))

plotting_params = {'plotting_resolution': 100}

# -------------------------------------------------------------------#

"""
Helper functions
"""

def dumpConstants(dict=system_constants, path=os.getcwd(), name='constants'):
    """Function that generates a json file from the dictionary provided.

    Args:
        dict (dict, optional): Dictionary to save. Defaults to system_constants.
        path (str, optional): Path to save json file to. Defaults to os.getcwd().
        name (str, optional): Name of file. Defaults to 'constants'.
    """
    _path = os.path.join(path, name+'.json')
    with open(_path, 'w') as c:
        json.dump(dict, c, indent=1)
        
def loadConstants(path='system_constants.json'):
    """Function that loads a json that was created with the dumpConstants function and writes the
    contents to a dictionary. 

    Args:
        path (str, optional): Path of the file to load. Defaults to 'system_constants.json'.

    Returns:
        dict: Dictionary containing the constants of the json file.
    """
    with open(path, 'r') as c:
        system_constants = json.load(c)
    return system_constants

# -------------------------------------------------------------------#
