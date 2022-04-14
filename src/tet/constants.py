import os
import json

constants = {'xA': None, 'xD': None, 'xMid': None,'max_N': None,\
    'max_t': None, 'omegaA': None, 'omegaMid': None,'omegaD': None,\
    'coupling': None, 'sites': None, 'resolution': None}

def setConstant(key, value):
    constants[key] = value
    
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