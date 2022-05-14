import numpy as np
import os
import shutil
import sys
import glob
from os.path import exists
import matplotlib.pyplot as plt
from saveFig import saveFig
import constants
from math import sqrt

# -------------------------------------------------------------------

"""
writeData: A function used for saving data in files
Documentation:
    * data: List/Array of data
    * destination: Path to save the file
    * name_of_file: Desired name of the file. Include the type of the file too.
"""

def writeData(data, destination, name_of_file):
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
#? TO DO 
def write_min_N(xA, xD, min_n, destination, name_of_file):
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

# -------------------------------------------------------------------

"""
read_1D_data: Assume that you are having a file with a float number per line.This function returns an array with that data.
Documentation:
    * destination: The path of the existing file:
    * name_of_file: The name of the file
"""
def read_1D_data(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    data = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        data.append(float(lines[0]))
    return data

# -------------------------------------------------------------------
"""
def read_2D_data(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    X,Y = [],[]
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        X.append(float(lines[0]))
        Y.append(float(lines[1]))
    return X,Y

def ReadDeque(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    to_return = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        to_return.append( (int(float(lines[0])),int(float(lines[1])),float(lines[2]),
                        int(float(lines[3])),int(float(lines[4]))) )

    return to_return 

def compress(zip_files, destination):
    if not zip_files: return None
    if zip_files:
        shutil.make_archive(base_name=f"{destination}-zipped", format='zip')
        shutil.rmtree(path=destination)
"""

# -------------------------------------------------------------------
"""
createDir: A function that creates a directory given the path
Documentation:
    * destination: The path to create the directory
    * replace_query: If true, permission is asked to overwrite the folder. If false, the directory gets replaced 
"""
def createDir(destination, replace_query=True):
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
                    shutil.rmtree(destination)
                    os.makedirs(destination)
                    break
            if fl_1 == 'n': sys.exit(0)
    else:
        os.makedirs(destination, exist_ok=True)

# -------------------------------------------------------------------

"""
PlotResults: A class for plotting results
Documentation
"""
class PlotResults:
    def __init__(self, const, data_path):
        self.const = const
        self.coupling = self.const['coupling']
        self.max_t = self.const['max_t']
        self.max_n = self.const['max_N']
        self.omegaD = self.const['omegas'][0]
        self.omegaA = self.const['omegas'][-1]
        self.sites = self.const['sites']
        self.Npoints = constants.plotting_params['plotting_resolution']

        self.data_path = data_path
        self.data_dirs = glob.glob(os.path.join(data_path, 'iteration_*'))

    # ONLY WORKS IN DIMER CASE
    def _plotHeatmap(self, data_path):

        lims = constants.lims
        init_chis = read_1D_data(data_path, 'init_chis.txt')
        xA = np.linspace(*lims[-1], num=self.Npoints)
        xD = np.linspace(*lims[0], num=self.Npoints)

        from tet.Execute import Execute
        data = Execute(chiA=xA, 
                       chiD=xD, 
                       coupling_lambda=self.coupling, 
                       omegaA=self.omegaA, 
                       omegaD=self.omegaD, 
                       max_N=self.max_n, 
                       max_t=self.max_t, 
                       data_dir=data_path,
                       return_data=True)()

        if np.ndim(data)>1:
            XA, XD = np.meshgrid(xA, xD)

            min_flat_data = np.zeros(len(xA)*len(xD))
            for i in enumerate(data): 
                min_flat_data[i[0]] = min(i[1])

            z = min_flat_data
            x = XD.flatten(order='C')
            y = XA.flatten(order='C')
            k = list(np.zeros(len(z)))
            for i in range(int(sqrt(len(min_flat_data)))):
                for j in range(int(sqrt(len(min_flat_data)))):
                    index = len(XA)*i+j
                    k[index] = x[index], y[index], z[index]
            
            min_n_combinations = np.array(k)

        xA_plot = min_n_combinations[:,0].reshape(self.Npoints, self.Npoints)
        xD_plot = min_n_combinations[:,1].reshape(self.Npoints, self.Npoints)
        avg_n = min_n_combinations[:,2].reshape(self.Npoints, self.Npoints)
        
        # Load Data
        a = read_1D_data(destination=data_path, name_of_file=f'x{self.sites-1}trajectory.txt')
        d = read_1D_data(destination=data_path, name_of_file=f'x{0}trajectory.txt')
        a_init = init_chis[-1]
        d_init = init_chis[0]
        
        # Plot heatmaps with optimizer predictions
        titl = f'N={self.max_n}, tmax={self.max_t}, Initial (χA, χD) = {a_init, d_init},\n\
            λ={self.coupling}, ωA={self.omegaA}, ωD={self.omegaD}'

        x = np.array(np.array(d))
        y = np.array(np.array(a))
        figure2, ax2 = plt.subplots(figsize=(12,12))
        plot2 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
        plot3 = ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
        plot4 = ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, label='Initial Value', zorder=3)
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot2)
        ax2.legend(prop={'size': 15})
        # ax2.set_title(titl, fontsize=20)
        saveFig(fig_id="contour", fig_extension="png", destination=data_path)
        plt.close(figure2)
    
    def plotHeatmap(self, all_opts=False, data_path=str()):
        if all_opts:
            for j, path in enumerate(self.data_dirs):
                iter_path = path
                opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
                for optimizer_i in opt_data:
                    self._plotHeatmap(optimizer_i)
        elif not all_opts and type(data_path)!=str:
            raise TypeError(f"Provided path variable is type {type(data_path).__name__}, not str.")
        else:
            self._plotHeatmap(data_path)

    def plotLoss(self):
        for j, path in enumerate(self.data_dirs):
            iter_path = path
            opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
            for optimizer_i in opt_data:
                loss_data = read_1D_data(destination=optimizer_i, name_of_file='losses.txt')
                # Plot Loss
                fig, ax = plt.subplots()
                ax.plot(loss_data[1:])
                saveFig(fig_id="loss", fig_extension="png", destination=optimizer_i, silent=True)
                plt.close(fig)
    
    # ONLY USABLE IN TRIMER AND DIMER CASE
    def plotScatterChis(self):
        
        for j, path in enumerate(self.data_dirs):
            iter_path = path
            opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
            optimal_vars = np.zeros((len(opt_data), self.sites+1))
            for i in range(optimal_vars.shape[0]):
                _chis = read_1D_data(destination=opt_data[i], name_of_file='optimalvars.txt')
                loss_data = read_1D_data(destination=opt_data[i], name_of_file='losses.txt')
                _loss = loss_data[-1]
                row = np.append(_chis, _loss)
                # Each row of optimal_vars is [*nonliniearity parameters, loss]
                optimal_vars[i,:] = row
            
            fig = plt.figure()
            if self.sites>2:
                ax = fig.add_subplot(projection='3d')
            else:
                ax = fig.add_subplot()
            # Scatterplot of final predicted parameters with colormap corresponding to the points' loss
            x = ax.scatter(*[optimal_vars[:,j] for j in range(optimal_vars.shape[1]-1)], c=optimal_vars[:,-1], cmap='plasma_r')
            fig.colorbar(x)
            saveFig(fig_id='chi_scatterplot', destination=path)
            plt.close(fig)

    def plotTimeEvol(self):

        import tensorflow as tf
        from HamiltonianLoss import Loss

        @tf.function
        def calcN(site):
            return l(*chis, single_value=False, site=site)

        for j, path in enumerate(self.data_dirs):
            iter_path = path
            opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
            for optimizer_i in opt_data:
                chis = read_1D_data(destination=optimizer_i, name_of_file='optimalvars.txt')
                self.const['chis'] = list(chis)
                l = Loss(const=self.const)
                evolved_n_acceptor = calcN(site=constants.acceptor)
                evolved_n_donor = calcN(site=constants.donor)
                fig, ax = plt.subplots()
                ax.plot(evolved_n_acceptor)
                ax.plot(evolved_n_donor)
                saveFig(fig_id="avg_n", fig_extension="png", destination=optimizer_i, silent=True)
                plt.close(fig)

if __name__=="__main__":

    data_paths = glob.glob(os.path.join(os.getcwd(), 'data_*'))
    data_params = [constants.loadConstants(path=os.path.join(path, 'constants.json')) for i, path in enumerate(data_paths)]
    for index, path in enumerate(data_paths):
        if data_params[index]['omegas'][1] == 2 and index>1:
            p = PlotResults(data_params[index], data_path=path)
            p.plotHeatmap(all_opts=True)

    # Test for 1 case
    # p = PlotResults(data_params[-1], data_path=data_paths[-1])
    # p.plotHeatmap(all_opts=False, data_path=r"C:\Users\Jason\source\repos\Thesis\data_1652461096118155600\iteration_1\data_optimizer_0")
