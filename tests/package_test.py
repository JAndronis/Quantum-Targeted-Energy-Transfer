import shutil
import sys
assert sys.version_info >= (3,6)
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import warnings

from tet import execute, saveFig, data_process

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=np.ComplexWarning)

    max_N = 12
    omegaA, omegaD = 3, -3
    # chiA, chiD = -0.5, 0.5
    coupling_lambda = 0.1
    t_max = 2000
    xA = np.linspace(-5, 5, 10)
    xD = np.linspace(-5, 5, 10)

    cwd = os.getcwd()
    data = f"{cwd}/data"

    coupling_dir = f"coupling-{coupling_lambda}"
    coupling_dir_path = os.path.join(data, coupling_dir)

    t_dir = f"tmax-{t_max}"
    t_dir_path = os.path.join(coupling_dir_path, t_dir)

    data_dest = os.path.join(t_dir_path, "avg_N")

    data_process.createDir(data)
    data_process.createDir(coupling_dir_path)
    data_process.createDir(t_dir_path)
    data_process.createDir(data_dest)

    zip_files = False
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
                            data_dir=data_dest)
    t2 = time.time()
    dt = t2-t1
    print(f"Code took: {dt:.3f}secs to run")

    t1 = time.time()
    execute.execute(chiA=xA, 
                    chiD=xD, 
                    coupling_lambda=coupling_lambda, 
                    omegaA=omegaA, 
                    omegaD=omegaD, 
                    max_N=max_N, 
                    max_t=t_max, 
                    data_dir=data_dest)
    t2 = time.time()
    dt = t2-t1
    print(f"Code took: {dt:.3f}secs to run")

    data_process.compress(zip_files=zip_files, destination=data_dest)
    if zip_files: sys.exit(0)

    data_analytical = []
    mimimums_ND = np.zeros(shape=(len(xA),len(xD)) )
    plt.figure(figsize=(8,8))
    for i in range(len(xA)):
        for j in range(len(xD)):
            title_analytical = f'ND_analytical-λ={coupling_lambda}-t_max={t_max}-χA={xA[i]}-χD={xD[j]}.txt'
            data_analytical_case = data_process.read_1D_data(destination=data_dest, name_of_file=title_analytical)
            mimimums_ND[i][j] = min(data_analytical_case)
            data_analytical.append([ data_analytical_case,xA[i],xD[j]])

    plt.imshow(mimimums_ND,cmap = 'gnuplot',extent=[min(xD),max(xD),max(xA),min(xA)])
    plt.xlabel('xD')
    plt.ylabel('xA')
    plt.colorbar()
    plt.title(f'tmax = {t_max}, points χA, χD = {len(xA), len(xD)}, λ={coupling_lambda}, ωA={omegaA}, ωD={omegaD}')
    title_heatmap = f'heatmap_tmax{t_max}_pointsxA:{len(xA)}_pointsxD{len(xD)}_λ={coupling_lambda}.pdf'
    saveFig.saveFig(title_heatmap, t_dir_path)