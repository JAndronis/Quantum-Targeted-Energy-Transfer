import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Execute import Execute
from itertools import product
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#Parameters of the problem
# --- omegaA = 3
# --- omgeaD = -3
# ---  N=12
# --- Coupling lambda = 1
#The goal is to find the optimal values of chiA,chiD

class ExtractPeriod:
    
    def __init__(self,xdata,ydata):
        self.xdata = xdata
        self.ydata = ydata
    
    def func(self, x, a, b, c,d):
        return a*np.sin(b*(x+c)) +d

    def EstimatePeriod(self):
        popt, _ = curve_fit(self.func, self.xdata, self.ydata)
        return 2*np.pi/popt[1]



class AverageProbability:

    def __init__(self,chiA_initial,chiD_initial,coupling_lambda,omegaA,omegaD,stepxA,stepxD,max_N,max_t):
        self.chiA_initial,self.chiD_initial = chiA_initial,chiD_initial
        self.coupling_lambda = coupling_lambda
        self.omegaA,self.omegaD = omegaA,omegaD
        self.max_N = max_N
        self.max_t = max_t
        self.stepxA = stepxA
        self.stepxD = stepxD


    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def AvgProb(self):
        #Changing the value of chiA,chiD
        chiAplus,chiAminus = self.chiA_initial + self.stepxA,self.chiA_initial - self.stepxA
        chiDplus,chiDminus = self.chiD_initial + self.stepxD,self.chiD_initial - self.stepxD

        avgND_chiAplus = Execute(chiAplus, self.chiD_initial,
                                 self.coupling_lambda,self.omegaA, self.omegaD,
                                 self.max_N,self.max_t, data_dir="", return_data=True).executeOnce()
        
        avgND_chiAminus = Execute(chiAminus, self.chiD_initial,
                                  self.coupling_lambda,self.omegaA, self.omegaD,
                                  self.max_N,self.max_t, data_dir="", return_data=True).executeOnce()
        
        avgND_chiDplus = Execute(self.chiA_initial,chiDplus,
                                 self.coupling_lambda,self.omegaA, self.omegaD,
                                 self.max_N,self.max_t, data_dir="", return_data=True).executeOnce()

        avgND_chiDminus = Execute(self.chiA_initial,chiDminus, 
                                  self.coupling_lambda,self.omegaA, self.omegaD,
                                  self.max_N,self.max_t, data_dir="", return_data=True).executeOnce()


        t_span = range(0,self.max_t+1)

        avgPD_chiAplus= np.array(avgND_chiAplus)/ self.max_N
        avgPD_chiAminus = np.array(avgND_chiAminus) / self.max_N
        avgPD_chiDplus= np.array(avgND_chiDplus)/ self.max_N
        avgPD_chiDminus = np.array(avgND_chiDminus) / self.max_N
        
        
        Period_chiAplus = ExtractPeriod(xdata=t_span,ydata = avgPD_chiAplus).EstimatePeriod()
        Period_chiAminus = ExtractPeriod(xdata=t_span,ydata = avgPD_chiAminus).EstimatePeriod()
        Period_chiDplus = ExtractPeriod(xdata=t_span,ydata = avgPD_chiDplus).EstimatePeriod()
        Period_chiDminus = ExtractPeriod(xdata=t_span,ydata = avgPD_chiDminus).EstimatePeriod()
        
        
        avgPD_chiAplus = np.average(avgPD_chiAplus[0:self.find_nearest(t_span,Period_chiAplus)] )
        avgPD_chiAminus = np.average(avgPD_chiAminus[0:self.find_nearest(t_span,Period_chiAminus)] )
        avgPD_chiDplus = np.average(avgPD_chiDplus[0:self.find_nearest(t_span,Period_chiDplus)] )
        avgPD_chiDminus = np.average(avgPD_chiDminus[0:self.find_nearest(t_span,Period_chiDminus)] )


        return avgPD_chiAplus,avgPD_chiAminus,avgPD_chiDplus,avgPD_chiDminus



class Reinforcement:


    def __init__(self,omegaD,omegaA,coupling_lambda,max_N,max_t,limitsxAxD,stepxA,stepxD,
                iterations,learning_rate):
        self.omegaD = omegaD
        self.omegaA = omegaA
        self.coupling_lambda = coupling_lambda
        self.max_N = max_N
        self.max_t = max_t
        self.min_xA,self.max_xA,self.min_xD,self.max_xD = limitsxAxD
        self.stepxA = stepxA
        self.stepxD = stepxD
        self.iterations = iterations
        self.learning_rate = learning_rate
        

    def An(self, i, xA, xD):
        f1 = self.omegaA + 0.5 * xA * (2*self.max_N - 2 * i - 1) - self.omegaD - 0.5 * xD * (2 * i + 1)
        return -tf.divide(tf.sqrt(float((i + 1) * (self.max_N - i))),f1)


    def Bn(self, i, xA, xD):
        f2 = -self.omegaA - 0.5 * xA * (2 * self.max_N - 2 * i + 1) + self.omegaD + 0.5 * xD * (2 * i - 1)
        return -tf.divide(tf.sqrt(float(i * (self.max_N - i + 1))),f2)


    def Cn2(self, i, xA, xD):
        f3 = 2*(self.omegaA - self.omegaD + xA * (self.max_N - i - 1) - xD * (i + 1))
        return tf.divide(self.An(i, xA, xD)**2*abs((i+2)*(self.max_N-i-1) )**2 , f3**2)
  

    def Dn(self, i, xA, xD):
        f4 = 2*(self.omegaD - self.omegaA - xA * (self.max_N - i + 1) + xD * (i - 1))
        return tf.divide(-self.Bn(i, xA, xD)*tf.sqrt(float((i-1)*(self.max_N-i+2))),f4)


    def Fn(self, i, xA, xD):
        return tf.divide(1,tf.sqrt(1+self.coupling_lambda**2*(tf.abs(self.An(i, xA, xD))**2 + tf.abs(self.Bn(i, xA, xD))**2)+self.coupling_lambda**4*(tf.abs(self.Cn2(i, xA, xD))+tf.abs(self.Dn(i, xA, xD))**2)))
    
    
    def PD(self, xA, xD):
        return self.Fn(self.max_N, xA, xD)**4*(1+self.coupling_lambda**2*self.Bn(self.max_N, xA, xD)**2+self.coupling_lambda**4*self.Dn(self.max_N, xA, xD)**2)

    """
    # ------- Step 1: Make an educated guess of the optimal values -------
    def InitialGuess(self):
        chiAs,chiDs = np.linspace(self.min_xA,self.max_xA,100),np.linspace(self.min_xD,self.max_xD,100)
        combinations = list(product(chiAs, chiDs))
        PDs = np.zeros(len(combinations))
        for i in range(len(combinations)):
            xA,xD = combinations[i]
            PDs[i] = self.PD(xA,xD).numpy()
        
        chiA_initial,chiD_initial = combinations[list(PDs).index(min(PDs))]

        return chiA_initial,chiD_initial
    """

    # ------- Step 2: RL ------- 
    def train(self):
        #chiA_initial,chiD_initial = self.InitialGuess()
        chiA_initial,chiD_initial = -0.3,0.3
        """        initial_trial_RL = Execute(chiA = chiA_initial, chiD = chiD_initial, 
                                   coupling_lambda = self.coupling_lambda,
                                   omegaA = self.omegaA, omegaD = self.omegaD,
                                   max_N = self.max_N, 
                                   max_t = 0,
                                   data_dir = "", return_data = True)
        """

        for _ in range(self.iterations):
            #Average Probability of being to donor
            avgPD_chiAplus,avgPD_chiAminus,avgPD_chiDplus,avgPD_chiDminus = AverageProbability(
                                                                         chiA_initial,chiD_initial,
                                                                         self.coupling_lambda,
                                                                         self.omegaA,
                                                                         self.omegaD,
                                                                         self.stepxA,self.stepxD,
                                                                         self.max_N,
                                                                         max_t = self.max_t ).AvgProb()
            
            der_xD = (0.5/self.stepxD)*(avgPD_chiDplus-avgPD_chiDminus)
            der_xA = (0.5/self.stepxA)*(avgPD_chiAplus-avgPD_chiAminus)

            chiA_initial += -self.learning_rate*der_xA
            chiD_initial += -self.learning_rate*der_xD

            print(chiA_initial,chiD_initial)



problem = Reinforcement(omegaD=-3,omegaA=3,
                        coupling_lambda=1,
                        max_N=12,
                        max_t = 3000,
                        limitsxAxD = [-2,2,-2,2],
                        stepxA=0.05,
                        stepxD = 0.05,
                        iterations = 3,
                        learning_rate = 0.05)
problem.train()
