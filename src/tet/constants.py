import os
import json
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
constants = {'max_N': 2,
             'max_t': 25, 
             'omegas': [-3, 3, 3],
             'chis': [0, 0, 0],
             'coupling': 1, 
             'sites': 3,
             'Npointst':100}

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
TensorflowParams = {'DTYPE': tf.float32, 
                    'lr': 0.1, 
                    'iterations': 500,
                    'tol':1e-8,
                    'train_sites': [0, 2]}

# Define the acceptor and the donor site
acceptor = 'x{}'.format(constants['sites']-1)
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
                'Npoints': 3,
                'epochs_grid':500,
                'epochs_bins':1000}

#! Create a dictionary with the limits of each variable explored
keys = [ f'x{i}lims' for i in TensorflowParams['train_sites'] ] 
lims = [[-5,5]]*len(keys)
TrainableVarsLimits = dict(zip(keys,lims))

plotting_params = {'plotting_resolution': 100}

# -------------------------------------------------------------------#

"""
Helper functions: You may ignore
"""

def setConstant(dict, key, value):
    dict[key] = value
    
def getConstant(key):
    return constants[key]

def dumpConstants(dict=constants, path=os.getcwd()):
    _path = os.path.join(path, 'constants.json')
    with open(_path, 'w') as c:
        #converts the Python objects into appropriate json objects
        json.dump(dict, c, indent=1)
        
def loadConstants(path='constants.json'):
    with open(path, 'r') as c:
        constants = json.load(c)
    return constants

# -------------------------------------------------------------------#

