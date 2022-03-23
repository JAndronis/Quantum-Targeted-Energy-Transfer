import os
import numpy as np
import matplotlib.pyplot as plt
from tet.Execute import Execute
from tet.data_process import createDir
from tet.saveFig import saveFig
from tet.data_process import write_min_N
import tet.constants as const

# set and save constants
const.setConstant('max_N', 3)
const.setConstant('max_t', 25)
const.setConstant('omegaA', 15)
const.setConstant('omegaD', -5)
const.setConstant('coupling', 0.1)
const.setConstant('resolution', 200)
const.dumpConstants()

CONST = const.loadConstants('constants.json')

def main():
    max_N = CONST['max_N']

    omegaA, omegaD = CONST['omegaA'], CONST['omegaD']
    coupling_lambda = CONST['coupling']
    t_max = CONST['max_t']
    
    xA = np.linspace(-10,10, CONST['resolution'])
    xD = np.linspace(-10,10, CONST['resolution'])
    
    write_data=True
    return_query = True

    cwd = os.getcwd()
    data = f"{cwd}/data"

    coupling_dir = f"coupling-{coupling_lambda}"
    coupling_dir_path = os.path.join(data, coupling_dir)

    t_dir = f"tmax-{t_max}"
    t_dir_path = os.path.join(coupling_dir_path, t_dir)

    data_dest = os.path.join(t_dir_path, "avg_N")

    if write_data:
        createDir(data, replace=False)
        createDir(coupling_dir_path, replace=False)
        createDir(t_dir_path)
        createDir(data_dest)
        
        test_data = Execute(chiA=xA, 
                                chiD=xD, 
                                coupling_lambda=coupling_lambda, 
                                omegaA=omegaA, 
                                omegaD=omegaD, 
                                max_N=max_N, 
                                max_t=t_max, 
                                data_dir=data_dest,
                                return_data=return_query)()
        const.dumpConstants(path=data)
    else:
        test_data = Execute(chiA=xA, 
                                chiD=xD, 
                                coupling_lambda=coupling_lambda, 
                                omegaA=omegaA, 
                                omegaD=omegaD, 
                                max_N=max_N, 
                                max_t=t_max, 
                                data_dir=os.getcwd(),
                                return_data=return_query)()
        
    if return_query and np.ndim(test_data)>1:
        XA, XD = np.meshgrid(xA, xD)

        counter = 0
        test_z = np.zeros(len(xA)*len(xD))
        for i in enumerate(test_data): 
            test_z[i[0]] = min(i[1])
            counter += 1
        test_z = test_z.reshape(len(xA), len(xD))

        write_min_N(xA=XA, xD=XD, min_n=test_z, destination=data_dest, name_of_file='min_n_combinations')
        titl = f'N={max_N}, tmax = {t_max}, # points (χA, χD) = {len(xA), len(xD)}, λ={coupling_lambda}, ωA={omegaA}, ωD={omegaD}'

        figure, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12,12))
        plot = ax.plot_surface(XA, XD, test_z, cmap='rainbow')
        ax.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure.colorbar(plot)
        ax.set_title(titl, fontsize=20)
        saveFig(titl+' - 3dplot', t_dir_path)
    
        figure2, ax2 = plt.subplots(figsize=(12,12))
        plot2 = ax2.contourf(test_z,cmap = 'rainbow',extent=[min(xD),max(xD),min(xA),max(xA)], levels=50)
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot2)
        ax2.set_title(f'N={max_N}, tmax = {t_max}, # points (χA, χD) = {len(xA), len(xD)},\nλ={coupling_lambda}, ωA={omegaA}, ωD={omegaD}', fontsize=20)
        saveFig(titl+' - contourplot', t_dir_path)
    else: print(np.min(test_data))

if __name__=="__main__":
    main()

