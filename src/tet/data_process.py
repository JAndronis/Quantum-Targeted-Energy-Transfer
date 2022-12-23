import numpy as np
import os
import sys
from os.path import exists


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
    else:
        fmt = '%.18e'

    if exists(_destination):
        # print('File exists, overwrite')
        with open(_destination, "a") as f:
            np.savetxt(f, data, fmt=fmt)

    else:
        np.savetxt(_destination, data, fmt=fmt)


# -------------------------------------------------------------------

def write_min_N(xa, xd, min_n, destination, name_of_file):
    """
    Function that is used to save data for plotting.

    Args:
        xa (np.ndarray): 2D array of xA parameters produced from np.meshgrid.
        xd (np.ndarray): 2D array of xd parameters produced from np.meshgrid.
        min_n (np.ndarray): 2D array of min_N bosons for every combination of xAs, xDs.
        destination (str): Where to save the file.
        name_of_file (str): How to name the file.
    """

    z = min_n.flatten(order='C')
    x = xd.flatten(order='C')
    y = xa.flatten(order='C')
    k = list(np.zeros(len(z)))

    for i in range(len(min_n)):
        for j in range(len(min_n)):
            index = len(xa) * i + j
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
                if query == '' or fl_1 not in ['y', 'n']:
                    print('Please answer with yes or no')
                else:
                    os.makedirs(destination)
                    break
            if fl_1 == 'n':
                sys.exit(0)
    else:
        os.makedirs(destination, exist_ok=True)
