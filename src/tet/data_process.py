import numpy as np
import os
import shutil
import sys
import glob
from os.path import exists
import matplotlib.pyplot as plt
from math import sqrt
import warnings

from . import constants
from .saveFig import saveFig

# -------------------------------------------------------------------#

def writeData(data, destination, name_of_file):
    """
    A function used for saving data in files

    Args:
        data: List/Array of data
        destination: Path to save the file
        name_of_file: Desired name of the file. Include the type of the file too.
    """
    data = np.array(data)
    _destination = os.path.join(destination, name_of_file)

    if data.dtype.name[:3] == 'str':
        fmt = '%s'
    else: fmt = '%.18e'
    
    if exists(_destination):
        #print('File exists, overwrite')
        with open(_destination, "a") as f:
            np.savetxt(f, data, fmt=fmt)
    
    else: np.savetxt(_destination, data, fmt=fmt)

# -------------------------------------------------------------------

def write_min_N(xA, xD, min_n, destination, name_of_file):
    """
    Function that is used to save data for plotting.

    Args:
        xA (np.ndarray): 2D array of xA parameters produced from np.meshgrid.
        xD (np.ndarray): 2D array of xD parameters produced from np.meshgrid.
        min_n (np.ndarray): 2D array of min_N bosons for every combination of xAs, xDs.
        destination (str): Where to save the file.
        name_of_file (str): How to name the file.
    """
    
    z = min_n.flatten(order='C')
    x = xD.flatten(order='C')
    y = xA.flatten(order='C')
    k = list(np.zeros(len(z)))
    index = 0

    for i in range(len(min_n)):
        for j in range(len(min_n)):
            index = len(xA)*i+j
            k[index] = x[index], y[index], z[index]
    
    temp_arr = np.array(k)
    writeData(data=temp_arr, destination=destination, name_of_file=name_of_file)

# -------------------------------------------------------------------#

def read_1D_data(destination, name_of_file):
    """
    Suppose you have a file with a float number per line.This function returns an array with that data.

    Args:
        destination (str): The path to save the file.
        name_of_file (str): The name of the file.
    """
    _destination = os.path.join(destination, name_of_file)
    data = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        data.append(float(lines[0]))
    return data

# -------------------------------------------------------------------#

def createDir(destination, replace_query=True):
    """
    A function that creates a directory given the path

    Args:
        destination (str): The path to create the directory
        replace_query (bool, optional): If true, permission is asked to overwrite the folder. If false, the directory gets replaced 
    """
    if replace_query:    
        try:
            os.mkdir(destination)
        except OSError as error:
            print(error)
            while True:
                query = input("Directory exists, replace it? [y/n] ")
                fl_1 = query[0].lower() 
                if query == '' or not fl_1 in ['y','n']: 
                    print('Please answer with yes or no')
                else:
                    os.makedirs(destination)
                    break
            if fl_1 == 'n': sys.exit(0)
    else:
        os.makedirs(destination, exist_ok=True)
