from cProfile import label
from math import comb
import sys

assert sys.version_info >= (3,6)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import time

from tet.data_process import createDir,writeData,read_1D_data
from tet.saveFig import saveFig
from itertools import combinations, product

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K

# enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# constants
DTYPE = tf.float32

LAMBDA = tf.constant(0.1, dtype=DTYPE)
OMEGA_A = tf.constant(3, dtype=DTYPE)
OMEGA_D = tf.constant(-3, dtype=DTYPE)
MAX_N = tf.constant(4, dtype=DTYPE)
MAX_T = tf.constant(25, dtype=tf.int32)
DIM = int(tf.constant(MAX_N+1).numpy())

OPT = tf.keras.optimizers.Adam(learning_rate=0.01)

class Loss:
    def __init__(self):
        self.coupling_lambda = LAMBDA
        self.omegaA = OMEGA_A
        self.omegaD = OMEGA_D
        self.max_N = MAX_N
        self.max_t = MAX_T
        self.dim = DIM

        initial_state = tf.TensorArray(DTYPE, size=self.dim)
        for n in range(self.dim):
            initial_state = initial_state.write(n, tf.exp(-tf.pow((self.max_N-n), 2)))
        self.initial_state = initial_state.stack()
        self.initial_state = tf.divide(self.initial_state, tf.linalg.norm(self.initial_state))

    def __call__(self, xA, xD):
        return self.loss(xA, xD)

    def createHamiltonian(self, xA, xD):
        h = tf.zeros((self.dim, self.dim), dtype=DTYPE)
        
        diag_indices = []
        upper_diag_indices = []
        lower_diag_indices = []
        
        diag_updates = []
        upper_diag_updates = []
        lower_diag_updates = []
        
        for i in range(self.dim):
            n = tf.cast(i, dtype=DTYPE)
            for j in range(self.dim):
                if i==j:
                    diag_indices.append([i,j])
                    diag_updates.append(self.omegaD * n + 0.5 * xD * n ** 2\
                            + self.omegaA * (self.max_N - n) + 0.5 * xA * (self.max_N - n) ** 2)
                if i==j-1:
                    lower_diag_indices.append([i,j])
                    lower_diag_updates.append(-self.coupling_lambda * tf.sqrt((n + 1) * (self.max_N - n)))
                if i==j+1:
                    upper_diag_indices.append([i,j])
                    upper_diag_updates.append(-self.coupling_lambda * tf.sqrt(n * (self.max_N - n + 1)))

        h = tf.tensor_scatter_nd_update(h, diag_indices, diag_updates)
        h = tf.tensor_scatter_nd_update(h, upper_diag_indices, upper_diag_updates)
        h = tf.tensor_scatter_nd_update(h, lower_diag_indices, lower_diag_updates)
        return h
    
    def coeffs(self, xA, xD):
        problemHamiltonian = self.createHamiltonian(xA, xD)
        eigvals, eigvecs = tf.linalg.eigh(problemHamiltonian)
        
        coeff_c = tf.TensorArray(DTYPE, size=self.dim)
        for i in range(self.dim):
            coeff_c = coeff_c.write(i, tf.tensordot(eigvecs[:,i], self.initial_state, 1))
        
        coeff_c = coeff_c.stack()
        coeff_b = eigvecs
        return coeff_c, coeff_b, eigvals
    
    def computeAverage(self, c, b, e):
        _time = MAX_T+1
        n = tf.TensorArray(DTYPE, size=_time)
        for t in range(_time):
            _t = tf.cast(t, dtype=tf.complex64)
            sum_j = tf.cast(0, dtype=tf.complex64)
            for j in range(self.dim):
                temp_b = tf.cast(b[j,:], dtype=tf.complex64)
                temp_c = tf.cast(c, dtype=tf.complex64)
                temp_e = tf.cast(e, dtype=tf.complex64)
                sum_i = tf.reduce_sum(temp_c*temp_b*tf.exp(-tf.complex(0.0,1.0)*temp_e*_t), 0)
                sum_k = tf.reduce_sum(temp_c*temp_b*tf.exp(tf.complex(0.0,1.0)*temp_e*_t)*sum_i, 0)
                j = tf.cast(j, dtype=tf.complex64)
                sum_j = sum_j+sum_k*j
            sum_j = tf.math.real(sum_j)
            n = n.write(t, sum_j)
        return n.stack()
        
    def loss(self, xA, xD):
        coeff_c, coeff_b, vals = self.coeffs(xA, xD)
        avg_N_list = self.computeAverage(coeff_c, coeff_b, vals)
        avg_N = tf.math.reduce_min(avg_N_list, name='Average_N')
        return avg_N



class Train:
    def __init__(self,ChiAInitial,ChiDInitial,DataExist,case):
        self.DataExist = DataExist
        self.ChiAInitial = ChiAInitial
        self.ChiDInitial = ChiDInitial
        self.data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
        self.CombinationPath = os.path.join(self.data_path,f'Combination {case}')

        #For the heatmap
        self.min_n_path = os.path.join(os.getcwd(), 'tests/data/coupling-'+str(LAMBDA.numpy())+'/tmax-'+
        str(MAX_T.numpy())+'/avg_N/min_n_combinations')
        #self.min_n_path = os.path.join(os.getcwd(), 'data/coupling-0.1/tmax-25/avg_N/min_n_combinations')
        self.test_array = np.loadtxt(self.min_n_path)
        self.xA_plot = self.test_array[:,0].reshape(POINTSBACKGROUND,POINTSBACKGROUND), 
        self.xD_plot = self.test_array[:,1].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
        self.avg_n = self.test_array[:,2].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
        

    def __call__(self):
        if self.DataExist:self.PlotResults()
        else:
            createDir(self.data_path, replace=False)
            createDir(destination=self.CombinationPath,replace=True)
            self.train()


    @tf.function
    def compute_loss(self,xA, xD):
        return Loss().loss(xA, xD)


    def get_grads(self,xA, xD):
        with tf.GradientTape() as t:
                t.watch([xA, xD])
                loss = self.compute_loss(xA, xD)
        grads = t.gradient(loss, [xA, xD])
        del t
        return grads, loss
        

    @tf.function
    def apply_grads(self,xA, xD):
        grads, loss = self.get_grads(xA, xD)
        OPT.apply_gradients(zip(grads, [xA, xD]))
        return loss


    def train(self):
        CHIA = tf.constant(self.ChiAInitial, dtype=DTYPE)
        CHID = tf.constant(self.ChiDInitial, dtype=DTYPE)
        mylosses = []
        tol = 1e-8
        max_iter = 1000
        xA = tf.Variable(initial_value=self.ChiAInitial, trainable=True, dtype=tf.float32)
        xD = tf.Variable(initial_value=self.ChiDInitial, trainable=True, dtype=tf.float32)
        xA_best = tf.Variable(initial_value=0, dtype=tf.float32)
        xD_best = tf.Variable(initial_value=0, dtype=tf.float32)
        mylosses.append(3.9)
        best_loss = MAX_N.numpy()
        counter = 0
        d_data = []
        a_data = []
        loss = Loss()

        t0 = time.time()
        for epoch in range(max_iter):
            xA_init = xA.numpy()
            xD_init = xD.numpy()
            loss = self.apply_grads(xA, xD)
            if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
            
            errorA = np.abs(xA.numpy() - xA_init)
            errorD = np.abs(xD.numpy() - xD_init)

            mylosses.append(loss.numpy())
            if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
                xA_best.assign(xA.numpy())
                xD_best.assign(xD.numpy())
                best_loss = mylosses[epoch+1]

            counter += 1
            if counter%10 == 0:
                d_data.append(xD.numpy())
                a_data.append(xA.numpy())

            if np.abs(loss.numpy()) < tol:
                break
            
            if errorA < tol:
                print('Stopped training because of xA_new-xA_old =', errorA)
                break

            if errorD < tol:
                print('Stopped training because of xD_new-xD_old =', errorA)
                break
            
        t1 = time.time()
        dt = t1-t0

        print("\nApproximate value of chiA:", xA_best.numpy(), 
            "\nApproximate value of chiD:", xD_best.numpy(),
            "\nLoss - min #bosons on donor:", best_loss,
            "\nOptimizer Iterations:", OPT.iterations.numpy(), 
            "\nTraining Time:", dt,
            "\n"+40*"-",
            "\nParameters:",
            "\nOmega_A:", OMEGA_A.numpy(),
            "| Omega_D:", OMEGA_D.numpy(),
            "| N:", MAX_N.numpy(),
            "| Total timesteps:", MAX_T.numpy(),
            "| Coupling Lambda:",LAMBDA.numpy(),
            "\n"+40*"-")

        writeData(data=mylosses[1:],destination=self.CombinationPath,name_of_file='losses.txt')
        writeData(data = a_data,destination=self.CombinationPath,name_of_file='xAs Trajectory.txt')
        writeData(data = d_data,destination=self.CombinationPath,name_of_file='xDs Trajectory.txt')
        writeData(data = [xA_best.numpy(),self.ChiAInitial],destination=self.CombinationPath,name_of_file='xAcharacteristics.txt')
        writeData(data = [xD_best.numpy(),self.ChiDInitial],destination=self.CombinationPath,name_of_file='xDcharacteristics.txt')
     

    def PlotResults(self):
        #Plot losses
        loss_data = read_1D_data(destination=self.CombinationPath,name_of_file='losses.txt')
        a= read_1D_data(destination=self.CombinationPath,name_of_file='xAs Trajectory.txt')
        d= read_1D_data(destination=self.CombinationPath,name_of_file='xAs Trajectory.txt')
        a_init = read_1D_data(destination=self.CombinationPath,name_of_file='xAcharacteristics.txt')[1]
        d_init = read_1D_data(destination=self.CombinationPath,name_of_file='xAcharacteristics.txt')[1]
        plt.figure()
        plt.plot(loss_data[1:])
        saveFig(fig_id="loss", destination=self.CombinationPath)
        plt.close()
        
        #Plot heatmaps
        
        titl = f'N={MAX_N.numpy()}, tmax={MAX_T.numpy()}, Initial (χA, χD) = {a_init, d_init}, λ={LAMBDA.numpy()}, ωA={OMEGA_A.numpy()}, ωD={OMEGA_D.numpy()}'    
        
        x = np.array(np.array(d))
        y = np.array(np.array(a))
        figure2, ax2 = plt.subplots(figsize=(12,12))
        # plot the predictions of the optimizer
        plot2 = ax2.contourf(self.xD_plot, self.xA_plot, self.avg_n, levels=50, cmap='rainbow')
        ax2.plot(x, y, marker='o', color='black', label='Optimizer Predictions')
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
        ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, label='Initial Value', zorder=3)
        ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
        ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
        figure2.colorbar(plot2)
        ax2.legend(prop={'size': 15})
        ax2.set_title(titl, fontsize=20)
        saveFig(fig_id="contour", destination=self.CombinationPath)



def MainGradient():
    TotalLosses,TotalMinLosses = [],[]
    TotalxATrajectories,TotalxDTrajectories = [],[]
    TotalxABest,TotalxDBest = [],[]

    #Read Data
    for case in range(len(Combinations)):
        data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
        CombinationPath = os.path.join(data_path,f'Combination {case}')
        loss_data = read_1D_data(destination=CombinationPath,name_of_file='losses.txt')
        xA_track = read_1D_data(destination=CombinationPath,name_of_file='xAs Trajectory.txt')
        xD_track = read_1D_data(destination=CombinationPath,name_of_file='xDs Trajectory.txt')
        xA_best = read_1D_data(destination=CombinationPath,name_of_file='xAcharacteristics.txt')[0]
        xD_best = read_1D_data(destination=CombinationPath,name_of_file='xDcharacteristics.txt')[0]
        TotalLosses.append(loss_data)
        TotalMinLosses.append(min(loss_data))
        TotalxATrajectories.append(xA_track)
        TotalxDTrajectories.append(xD_track)
        TotalxABest.append(xA_best)
        TotalxDBest.append(xD_best)
    
    #Find the minimum loss
    IndexLowestLoss = TotalMinLosses.index(min(TotalMinLosses))
    #Begin Training based on that result
    xAInitMainGradient,xDInitMainGradient = TotalxABest[IndexLowestLoss],TotalxDBest[IndexLowestLoss]
    print('-'*35 + 'Begin Main Gradient' + '-'*35)
    Train(ChiAInitial=xAInitMainGradient,ChiDInitial=xDInitMainGradient,
                DataExist=False,
                case = NPointsxA*NPointsxD)()



def PlotMainGradientData():
    #Plot the initial heatmap
    success_indices = []
    min_n_path = os.path.join(os.getcwd(), 'tests/data/coupling-'+str(LAMBDA.numpy())+'/tmax-'+
        str(MAX_T.numpy())+'/avg_N/min_n_combinations')
    #min_n_path = os.path.join(os.getcwd(), 'data/coupling-0.1/tmax-25/avg_N/min_n_combinations')
    test_array = np.loadtxt(min_n_path)
    xA_plot= test_array[:,0].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
    xD_plot = test_array[:,1].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
    avg_n =test_array[:,2].reshape(POINTSBACKGROUND,POINTSBACKGROUND)
    figure2, ax2 = plt.subplots(figsize=(12,12))
    # plot the predictions of the optimizer
    plot2 = ax2.contourf(xD_plot, xA_plot, avg_n, levels=50, cmap='rainbow')
    #Load data from test agents
    for case in range(len(Combinations)):
        data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
        CombinationPath = os.path.join(data_path,f'Combination {case}')
        loss_data = read_1D_data(destination=CombinationPath,name_of_file='losses.txt')
        if min(loss_data) < 0.3:success_indices.append(case)
    
    for case in success_indices:
        data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
        CombinationPath = os.path.join(data_path,f'Combination {case}')
        xA_track = read_1D_data(destination=CombinationPath,name_of_file='xAs Trajectory.txt')
        xD_track = read_1D_data(destination=CombinationPath,name_of_file='xDs Trajectory.txt')
        d_init,a_init = Combinations[case][1],Combinations[case][0]
        x = np.array(np.array(xD_track))
        y = np.array(np.array(xA_track))
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2)
        ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
        ax2.scatter(d_init, a_init, color='green', edgecolors='black', s=94, zorder=3)
        
    #Just to include the label, these 2 lines are useless
    d_initplot,a_initplot = Combinations[success_indices[0]][1],Combinations[success_indices[0]][0]
    ax2.scatter(d_initplot, a_initplot, color='green', edgecolors='black',label = 'Initial Guesses of Test Agents', s=94, zorder=3)
    #Load data from the main agent
    data_path = os.path.join(os.getcwd(), 'data_optimizer_avgn')
    CombinationPath = os.path.join(data_path,f'Combination {NPointsxA*NPointsxD}')
    xA_track = read_1D_data(destination=CombinationPath,name_of_file='xAs Trajectory.txt')
    xD_track = read_1D_data(destination=CombinationPath,name_of_file='xDs Trajectory.txt')
    d_init = read_1D_data(destination=CombinationPath,name_of_file='xDcharacteristics.txt')[1]
    a_init = read_1D_data(destination=CombinationPath,name_of_file='xAcharacteristics.txt')[1]

    x = np.array(np.array(xD_track))
    y = np.array(np.array(xA_track))
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2)
    ax2.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy",pivot="mid")
    ax2.scatter(d_init, a_init, color='b', edgecolors='black',label='Initial Guess of Main Agent', s=94, zorder=3)

    # title and etc
    ax2.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax2.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure2.colorbar(plot2)
    ax2.legend(prop={'size': 15})
    titl = 'N={}, tmax={}, λ={:.3f}, ωA={}, ωD={},#Test Agents = {}'.format(

    int(MAX_N.numpy()),MAX_T.numpy(),LAMBDA.numpy(),OMEGA_A.numpy(),OMEGA_D.numpy(),NPointsxA*NPointsxD) 
    #set_labels(ax2.plot(x, y), 'ABC')
    ax2.set_title(titl, fontsize=20)

    FinalPath = os.path.join(data_path,'Final Data')
    createDir(destination=FinalPath,replace=True)
    saveFig(fig_id="FinalContour", destination=FinalPath)

def PlotFinalPlot():
    xAsoptimal,xDsoptimal =[],[]
    Ns = []
    #Read data
    for i in range(2,14):
        _case = f'Case {i}'
        # PathToMeeting = r"C:\Users\Giwrgos Arapantonis\Desktop\8th semester\Meetings\March\March 8th"
        # PathToCase = os.path.join(PathToMeeting,_case)
        PathToOptimizerData = os.path.join(os.getcwd(),'data_optimizer_avgn')
        #Number of folders in this directory
        NDirs = len(os.listdir(PathToOptimizerData))
        #Index to main agent
        IndexMainAgent = NDirs-2
        TitleMainAgent = f'Combination {IndexMainAgent}'
        PathMainAgent = os.path.join(PathToOptimizerData,TitleMainAgent)

        #Go to data
        xD_optimal = read_1D_data(destination=PathMainAgent,name_of_file='xDcharacteristics.txt')[0]
        xA_optimal = read_1D_data(destination=PathMainAgent,name_of_file='xAcharacteristics.txt')[0]

        xAsoptimal.append(xA_optimal)
        xDsoptimal.append(xD_optimal)
        Ns.append(i)
        

    plt.figure(figsize=(15,4))
    Nsample = np.linspace(1,13,50)
    xDsample = 6/Nsample
    xAsample = -xDsample
    #N-xD
    plt.subplot(1,2,1)
    plt.plot(Nsample,xDsample,label=r"Theory: $\frac{\omega_A-\omega_D}{N}$")
    plt.scatter(Ns,xDsoptimal,label='Experimental',c='r')
    plt.xlabel("Number of bosons")
    plt.ylabel(r"$\chi_{D}$")
    plt.legend()

    #N-xA
    plt.subplot(1,2,2)
    plt.plot(Nsample,xAsample,label=r"Theory: $\frac{\omega_A-\omega_D}{N}$")
    plt.scatter(Ns,xAsoptimal,label='Experimental',c='r')
    plt.xlabel("Number of bosons")
    plt.ylabel(r"$\chi_{A}$")
    plt.legend()
    
    plt.savefig('N-ChiAChiD.pdf')
    plt.close()

    plt.figure()
    #xA-xD
    alphas = np.linspace(1, 0.1, len(xAsoptimal))
    for i in range(len(xAsoptimal)):
        plt.scatter(xAsoptimal[i],xDsoptimal[i],alpha=alphas[i],c='b')
    plt.scatter(xAsoptimal[0],xDsoptimal[0],alpha=alphas[0],c='b',label=f'N={Ns[0]}')
    plt.scatter(xAsoptimal[len(xAsoptimal)-1],xDsoptimal[len(xAsoptimal)-1],alpha=alphas[len(xAsoptimal)-1],
    c='b',label=f'N={Ns[len(xAsoptimal)-1]}')
    plt.xlabel(r"$\chi_{A}$")
    plt.ylabel(r"$\chi_{D}$")
    plt.legend()
    plt.savefig('ChiA-ChiD.pdf')
    plt.close()    
    


if __name__=="__main__":
    NPointsxA = 4
    NPointsxD = 4
    POINTSBACKGROUND = 250
    ChiAInitials= np.linspace(-3,3,NPointsxA)
    ChiDInitials= np.linspace(-3,3,NPointsxD)
    Combinations = list(product(ChiAInitials,ChiDInitials))
    DATAEXIST,MAINGRADIENT= True,True
    
    
    if not MAINGRADIENT:
        for index,(ChiAInitial,ChiDInitial) in enumerate(Combinations):
            print('-'*20+'Combination:{} out of {},Initials (xA,xD):({:.3f},{:.3f})'.format(index,len(Combinations)-1,ChiAInitial,ChiDInitial) + '-'*20)
            #print(index)
            Train(ChiAInitial=ChiAInitial,ChiDInitial=ChiDInitial,
                DataExist=DATAEXIST,
                case = index)()
       
    else: MainGradient()
    
    

    PlotMainGradientData()
    # PlotFinalPlot()



