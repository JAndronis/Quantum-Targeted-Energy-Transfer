import os
import json
from tensorflow import float32

# -------------------------------------------------------------------#

"""
system_constants: A dictionary that defines the constant parameters of the problem

Elements:
    max_N: The number of bosons belonging to the system
    sites: The number of the system's oscillators. Usually denoted by f
    max_t: Period of time to compute the loss function
    omegas: An f-dimensional list with the frequency of the oscillators of the system
    chis: An f-dimensional list with the nonlinearity parameters of the oscillators
    Include the constant value of the non trainable parameters and enter random values for the trainable ones
    coupling: The characteristic coupling parameter
    timesteps: Determines the steps in time for computing the loss function.
    
"""
system_constants = {'max_N': 3,
             'sites': 3,
             'max_t': 25,
             'sites': 3, 
             'omegas': [-3,3,3],
             'chis': [1.5,0,-1.5],
             'coupling': 1,
             'timesteps': 30}

# -------------------------------------------------------------------#

"""
TensorflowParams: A dictionary that includes the parameters used for the commands of the Tensorflow library

Elements:
    DTYPE: dtype of parameters
    lr: The default value of the learning rate of each optimizer
    iterations: The default number of maximum Iterations of each optimizer
    tol: The tolerance of each optimizer concerning the changes in the nonlinearity parameters
    train_sites: A list including the nonlinearity parameters to be optimized. Counting ranges from 0 to f-1.
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

Elements:
    methods: Must remain immutable. A list with the possible methods of setting the initial guesses of the optimizers.
    target: Taking values of the format xk, where k is integer ranging from 0 to f-1. 
    It determines the site of which you desire to compute the loss function.Default value is x{f-1}, 
    referring to the acceptor. 
    Npoints: The num parameter in np.linspace. Given the limits of a trainable parameter, it determines the desnsity of the 1D grid.
    epochs_grid: The maximum iterations of each optimizer when using the grid method
    epochs_bins: The maximum iterations of each optimizer when using the bins method

"""
solver_params = {'methods': ['grid', 'bins'],
                'target': acceptor,
                'Npoints': 4,
                'epochs_grid':500,
                'epochs_bins':1000}



# -------------------------------------------------------------------#
"""
Create a dictionary with the limits of each trainable nonlinearity parameter.
"""
keys = [f'x{i}lims' for i in TensorflowParams['train_sites']] 
lims = [[-10,10]]*len(keys)
TrainableVarsLimits = dict(zip(keys,lims))


# -------------------------------------------------------------------#
"""
Defining the resolution of the figures created
"""
plotting_params = {'plotting_resolution': 100}


# -------------------------------------------------------------------#

def dumpConstants(dict: dict, path=os.getcwd(), name='constants'):
    """
    Function that generates a json file from the dictionary provided.

    Args:
        dict (dict): Dictionary to save.
        path (str, optional): Path to save json file to. Defaults to os.getcwd().
        name (str, optional): Name of file. Defaults to 'constants'.
    """
    _path = os.path.join(path, name+'.json')
    with open(_path, 'w') as c:
        json.dump(dict, c, indent=1)

# -------------------------------------------------------------------#

def loadConstants(path='system_constants.json') -> dict:
    """ 
    Function that loads a json that was created with the dumpConstants function and writes the
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
