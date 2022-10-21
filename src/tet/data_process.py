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

# -------------------------------------------------------------------

"""
writeData: A function used for saving data in files
"""
def writeData(data, destination, name_of_file):
    """
    Documentation:
        * data: List/Array of data
        * destination: Path to save the file
        * name_of_file: Desired name of the file. Include the type of the file too.
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
    """_summary_

    Args:
        * xA (_type_): _description_
        * xD (_type_): _description_
        * min_n (_type_): _description_
        * destination (_type_): _description_
        * name_of_file (_type_): _description_
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

# -------------------------------------------------------------------

"""
read_1D_data: Suppose you have a file with a float number per line.This function returns an array with that data.
"""
def read_1D_data(destination, name_of_file):
    """
    Documentation:
        * destination: The path of the existing file
        * name_of_file: The name of the file
    """
    _destination = os.path.join(destination, name_of_file)
    data = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        data.append(float(lines[0]))
    return data

# -------------------------------------------------------------------

"""
createDir: A function that creates a directory given the path
"""
def createDir(destination, replace_query=True):
    """
    Documentation:
        * destination: The path to create the directory
        * replace_query: If true, permission is asked to overwrite the folder. If false, the directory gets replaced 
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
                    shutil.rmtree(destination)
                    os.makedirs(destination)
                    break
            if fl_1 == 'n': sys.exit(0)
    else:
        os.makedirs(destination, exist_ok=True)

# -------------------------------------------------------------------

class PlotResults:
    """
    A class for plotting results

    Args:
        * const (dict): Dictionary of the problem parameters of like the constants.constants var
        * data_path (str):  Path to data directory
    """
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
    def plotHeatmap(self, all_opts=False, data_path=str(), ax=None, return_ax=False):

        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s}"

        lims = [[-8, 8], [-8, 8]]
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

        if all_opts:
            if ax == None:
                fig = plt.figure()
                ax = fig.add_subplot()
            plot = ax.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='YlGn_r')  # change cmap
            plot2 = ax.contour(xD_plot, xA_plot, avg_n, levels=5, colors=('k',))
            ax.clabel(plot2, plot2.levels, inline=True, fmt=fmt)
            counter=0

            for j, path in enumerate(self.data_dirs):
                iter_path = path
                opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
                
                for optimizer_i in opt_data:
                    if min(read_1D_data(destination=optimizer_i, name_of_file='losses.txt')) < 0.5:
                        init_chis = read_1D_data(optimizer_i, 'init_chis.txt')
                        # Load Data
                        a = read_1D_data(destination=optimizer_i, name_of_file=f'x{self.sites-1}trajectory.txt')
                        d = read_1D_data(destination=optimizer_i, name_of_file=f'x{0}trajectory.txt')
                        a_init = init_chis[-1]
                        d_init = init_chis[0]
                            
                        x = np.array(d)
                        y = np.array(a)
                        plot3 = ax.plot(x, y, marker='.', color='black', label='Test Opt. Predictions' if counter == 0 else '')
                        u = np.diff(x)
                        v = np.diff(y)
                        pos_x = x[:-1] + u/2
                        pos_y = y[:-1] + v/2
                        norm = np.sqrt(u**2+v**2)
                        plot4 = ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy")
                        plot5 = ax.scatter(d_init, a_init, color='cornflowerblue', edgecolors='black', s=94, label='Test Opt. Initial Guesses' if counter == 0 else '', zorder=3)
                        counter += 1
            
            chis = constants.loadConstants(path=os.path.join(self.data_path, 'constants.json'))['chis']
            plot5 = ax.scatter(*chis, color='indianred', edgecolors='black', s=94, label='Optimal Parameters', zorder=3)
            ax.set_xlabel(r"$\chi_{D}$")
            ax.set_ylabel(r"$\chi_{A}$")
            ax.legend()
            # cbar = figure2.colorbar(plot)
            # cbar.set_label('Loss Value')
            # saveFig(fig_id="contour", fig_extension="png", destination=self.data_path)
            ax.annotate("(a)", xycoords='axes points', xy=(-30,-26), fontweight='bold', fontsize=14)
            if return_ax:
                return ax, plot
            
        elif not all_opts and type(data_path)!=str:
            raise TypeError(f"Provided path variable is type {type(data_path).__name__}, not str.")
        
        else:
            figure2, ax = plt.subplots()
            plot = ax.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='YlGn_r')
            plot2 = ax.contour(xD_plot, xA_plot, avg_n, levels=5, colors=('k',))
            ax.clabel(plot2, plot2.levels, inline=True, fmt=fmt)

            if min(read_1D_data(destination=data_path, name_of_file='losses.txt')) < 0.5:
                init_chis = read_1D_data(data_path, 'init_chis.txt')
                # Load Data
                a = read_1D_data(destination=data_path, name_of_file=f'x{self.sites-1}trajectory.txt')
                d = read_1D_data(destination=data_path, name_of_file=f'x{0}trajectory.txt')
                a_init = init_chis[-1]
                d_init = init_chis[0]
                x = np.array(np.array(d))
                y = np.array(np.array(a))
                # plot3 = ax.plot(x, y, marker='o', color='black', label='Optimizer Predictions' if i == 0 else '')
                u = np.diff(x)
                v = np.diff(y)
                pos_x = x[:-1] + u/2
                pos_y = y[:-1] + v/2
                norm = np.sqrt(u**2+v**2)
                plot4 = ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", alpha=0.5, label='Optimizer Trajectory')
                plot5 = ax.scatter(d_init, a_init, color='indianred', edgecolors='black', s=94, label='Initial Value', zorder=3)
                ax.set_xlabel(r"$\chi_{D}$")
                ax.set_ylabel(r"$\chi_{A}$")
                    
            ax.legend()
            cbar = figure2.colorbar(plot)
            cbar.set_label('Min Loss Value')
            saveFig(fig_id="contour", fig_extension="png", destination=data_path)

            if return_ax:
                return ax, plot

    def plotLoss(self, save_path=None):

        for j, path in enumerate(self.data_dirs):
            iter_path = path
            opt_data = glob.glob(os.path.join(iter_path, 'data_optimizer_*'))
            for optimizer_i in opt_data:
                loss_data = read_1D_data(destination=optimizer_i, name_of_file='losses.txt')
                # Plot Loss
                fig, ax = plt.subplots()
                ax.plot(loss_data[1:])
                if save_path == None:
                    save_path = path
                saveFig(fig_id="loss", fig_extension="png", destination=save_path, silent=True)
                plt.close(fig)
    
    # ONLY USABLE IN TRIMER AND DIMER CASE
    def plotScatterChis(self, save_path=None):
        
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
            x = ax.scatter(*[optimal_vars[:,j] for j in range(optimal_vars.shape[1]-1)], c=optimal_vars[:,-1], cmap='YlGn_r')
            fig.colorbar(x)

            if save_path == None:
                    save_path = path
            saveFig(fig_id='chi_scatterplot', destination=save_path)
            plt.close(fig)

    def plotTimeEvol(self, save_path=None, main_opt=False):

        import tensorflow as tf
        from HamiltonianLoss import Loss

        steps = self.const['timesteps']
        t = np.linspace(0, self.const['max_t'], steps)

        @tf.function
        def calcN(site):
            return l(*chis, single_value=False, site=site)

        if not main_opt:
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
                    ax.plot(t, evolved_n_acceptor)
                    ax.plot(t, evolved_n_donor)

                    if save_path == None:
                        save_path = path

                    saveFig(fig_id="avg_n", fig_extension="png", destination=save_path, silent=True)
                    plt.close(fig)
        else:
            chis = read_1D_data(destination=os.path.join(self.data_path, 'main_opt'), name_of_file='optimalvars.txt')
            self.const['chis'] = list(chis)
            l = Loss(const=self.const)
            evolved_n_acceptor = calcN(site=constants.acceptor)
            evolved_n_donor = calcN(site=constants.donor)

            fig, ax = plt.subplots(figsize=(3+3/8, 2.8))
            ax.plot(t, evolved_n_acceptor, linewidth=2, label=r'$\langle N_A(t) \rangle$', color='indianred')
            ax.plot(t, evolved_n_donor, linewidth=2, label=r'$\langle N_D(t) \rangle$', color='cornflowerblue')
            ax.set_xlabel('t')
            ax.set_ylabel(r'$\langle N(t)\rangle$')
            ax.legend(loc=1, prop={'size': 8})
            ax.grid(linestyle='--', zorder=1)

            if save_path == None:
                save_path = os.path.join(self.data_path, 'main_opt')

            saveFig(fig_id="avg_n", fig_extension="png", destination=save_path, silent=True)
            plt.close(fig)

# MAIN ------------------------------------------------------------------------------- #

if __name__=="__main__":
    import constants

    data_paths = glob.glob(os.path.join(os.getcwd(), 'data_*'))
    data_params = [constants.loadConstants(path=os.path.join(path, 'constants.json')) for i, path in enumerate(data_paths)]
    ndata = [data_params[i]['max_N'] for i in range(len(data_params))]
    nlabels = [f'N={i}' for i in ndata]
    loss_data = [data_params[i]['min_n'] for i in range(len(data_params))]
    chi_as = [data_params[i]['chis'][-1] for i in range(len(data_params))]
    chi_ds = [data_params[i]['chis'][0] for i in range(len(data_params))]

    fig0, ax0 = plt.subplots()
    ax0.scatter(ndata, loss_data)
    plt.show()
    plt.close(fig0)

    ax = [0, 0, 0]

    fig = plt.figure(figsize=(2*(3+3/8), 5))
    spec = fig.add_gridspec(4, 6)
    ax[0] = fig.add_subplot(spec[:, :4])
    ax[1] = fig.add_subplot(spec[:2,4:])
    ax[2] = fig.add_subplot(spec[2:,4:])
    figure1_fontsize = 12
    xaxdplot = ax[1].scatter(chi_as, chi_ds, color='indianred', edgecolors='black', zorder=2)

    ax[1].annotate("",
                xy=(chi_as[0], chi_ds[0]), xycoords='data',
                xytext=(-4.5, 5.5), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="k",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3, rad=0.05",
                                ),
                )
    ax[1].text(-4.5, 5.5, nlabels[0], fontsize=8)

    ax[1].annotate("",
                xy=(chi_as[1], chi_ds[1]), xycoords='data',
                xytext=(-3.7, 3.9), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="k",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3, rad=0.05",
                                ),
                )
    ax[1].text(-4, 4, nlabels[1], fontsize=8)

    ax[1].annotate("",
                xy=(chi_as[2], chi_ds[2]), xycoords='data',
                xytext=(-2, 3), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="k",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3, rad=0.05",
                                ),
                )
    ax[1].text(-2, 3, nlabels[2], fontsize=8)

    ax[1].annotate("",
                xy=(chi_as[3], chi_ds[3]), xycoords='data',
                xytext=(-1.5, 2.3), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="k",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3, rad=0.05",
                                ),
                )
    ax[1].text(-1.5, 2.3, nlabels[3], fontsize=8)

    xtext, ytext = -3.8, 2
    counter = 1
    for label, x, y in zip(nlabels[4:], chi_as[4:], chi_ds[4:]):
        ax[1].annotate("",
                    xy=(x, y), xycoords='data',
                    xytext=(xtext+0.9, ytext+0.2), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="k",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc, rad=0.01",
                                    ),
                    )
        ax[1].text(xtext, ytext, label, fontsize=8)
        ytext -= 0.32
        counter += 1
    ax[1].annotate("(b)", xycoords='axes points', xy=(-25,-17), fontweight='bold', fontsize=14)
    ax[1].set_xlabel(rf'$\chi_A$')
    ax[1].set_ylabel(rf'$\chi_D$')
    ax[1].grid(linestyle='--', zorder=1)

    omegas_diff = [(data_params[i]['omegas'][-1] - data_params[i]['omegas'][0])/ data_params[i]['max_N'] for i in range(len(data_params))]

    from scipy.interpolate import interp1d

    f = interp1d(ndata, omegas_diff, kind='quadratic')

    omegas_diff_over_n = ax[2].plot(ndata, f(ndata), zorder=2, color='k', linestyle='--', label='Theory')
    predicted_chis = ax[2].scatter(ndata, chi_ds, color='indianred', edgecolors='k', label='Predicted', zorder=3)
    ax[2].set_xlabel(r'$N$')
    ax[2].set_ylabel(r'$(\omega_A - \omega_D)\cdot(N^{-1})$')
    ax[2].grid(linestyle='--', zorder=1)
    ax[2].annotate("(c)", xycoords='axes points', xy=(-25,-17), fontweight='bold', fontsize=14)
    ax[2].legend(prop={'size': 8})

    # saveFig(fig_id="chiad", fig_extension="png", destination=os.getcwd(), silent=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p = PlotResults(data_params[2], data_path=data_paths[2])
        _, heatmap = p.plotHeatmap(all_opts=True, ax=ax[0], return_ax=True)
    
    cbar = fig.colorbar(heatmap, ax=ax[0])
    cbar.set_label('Loss Value')
    saveFig(fig_id="chiad", fig_extension="png", destination=os.getcwd(), silent=False)

    p.plotTimeEvol(main_opt=True, save_path=os.getcwd())
