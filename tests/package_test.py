import shutil
import sys
assert sys.version_info >= (3,6)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings

from tet import execute, saveFig, writeData

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=np.ComplexWarning)

    max_N = 12
    omegaA, omegaD = 2, -3
    # chiA, chiD = -0.5, 0.5
    coupling_lambda = 1
    t_max = 200
    xA = np.linspace(-4, 4, 10)
    xD = np.linspace(-4, 4, 10)

    cwd = os.getcwd()
    new_data = f"{cwd}/new_data"

    try:
        os.mkdir(new_data)
    except OSError as error:
        print(error)
        while True:
            query = input("Directory exists, replace it? [y/n] ")
            fl_1 = query[0].lower() 
            if query == '' or not fl_1 in ['y','n']: 
                print('Please answer with yes or no')
            else: break
        if fl_1 == 'n': sys.exit(0)

    first_dir = f"coupling-{coupling_lambda}"
    first_dir_dest = os.path.join(new_data, first_dir)
    try:
        os.mkdir(first_dir_dest)
    except OSError as error:
        print(error)
        while True:
            query = input("Directory exists, replace it? [y/n] ")
            fl_1 = query[0].lower() 
            if query == '' or not fl_1 in ['y','n']: 
                print('Please answer with yes or no')
            else: break
        if fl_1 == 'n': sys.exit(0)

    dir_name = f"tmax-{t_max}"
    data_dest = os.path.join(first_dir_dest, dir_name)
    try:
        os.mkdir(data_dest)
    except OSError as error:
        print(error)
        while True:
            query = input("Directory exists, replace it? [y/n] ")
            fl_1 = query[0].lower() 
            if query == '' or not fl_1 in ['y','n']: 
                print('Please answer with yes or no')
            else: break
        if fl_1 == 'n': sys.exit(0)

    zip_files = True
    count_it = 0

    t1 = time.time()
    for chiA in xA:
        for chiD in xD:
            count_it +=1
            print("\rCombination {} out of {}: (chiA,chiD) = ({},{})".format(count_it,len(xA)*len(xD),round(chiA,4),round(chiD,4)), end = " ")
            execute.execute(chiA=chiA, 
                            chiD=chiD, 
                            coupling_lambda=coupling_lambda, 
                            omegaA=omegaA, 
                            omegaD=omegaD, 
                            max_N=max_N, 
                            max_t=t_max, 
                            data_dir=data_dest, 
                            zip_files=zip_files)
    t2 = time.time()
    dt = t2-t1
    print(f"Code took: {dt:.3f}secs to run")

    writeData.compress(zip_files=zip_files, destination=data_dest)
    if zip_files: sys.exit(0)

    data_analytical = []
    mimimums_ND = np.zeros(shape=(len(xA),len(xD)) )
    # t_span = np.linspace(0,t_max,t_max+1)
    plt.figure(figsize=(8,8))
    for i in range(len(xA)):
        for j in range(len(xD)):
            title_analytical = f'ND_analytical-λ={coupling_lambda}-t_max={t_max}-χA={xA[i]}-χD={xD[j]}.txt'
            data_analytical_case = writeData.read_1D_data(destination=data_dest, name_of_file=title_analytical)
            mimimums_ND[i][j] = min(data_analytical_case)
            data_analytical.append([ data_analytical_case,xA[i],xD[j]])

    plt.imshow(mimimums_ND,cmap = 'gnuplot',extent=[min(xD),max(xD),max(xA),min(xA)])
    plt.xlabel('xD')
    plt.ylabel('xA')
    plt.colorbar()
    plt.title(f'tmax = {t_max}, points χA, χD = {len(xA), len(xD)}, λ={coupling_lambda}, ωA={omegaA}, ωD={omegaD}')
    title_heatmap = f'heatmap_tmax{t_max}_pointsxA:{len(xA)}_pointsxD{len(xD)}_λ={coupling_lambda}.pdf'
    saveFig.saveFig(title_heatmap)