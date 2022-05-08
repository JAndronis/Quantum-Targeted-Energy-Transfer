import os
import json
import tensorflow as tf
#
constants = {'max_N': 4,
             'max_t': 25, 
             'omegas': [-3, 3],
             'chis': [.5, -.5],
             'coupling': 0.1, 
             'sites': 2, 
             }

# Parameters of tensorflow
TensorflowParams = {'DTYPE': tf.float32, 
                    'lr': 0.1, 
                    'iterations': 200,
                    'tol':1e-8,
                    }

# Solver Parameters
solver_params = {'methods': ['grid', 'bins'],
                'target': 'x{}'.format(constants['sites']-1),
                'Npoints': 5,
                'epochs_grid':200,
                'epochs_bins':1000}

# -------------- Helper Functions -------------- #

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
