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
"""
#2ND EDITION
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Execute import Execute
from itertools import product
from matplotlib import pyplot as plt
from scipy.fft import fftfreq
from scipy.fft import fft

#Parameters of the problem
# --- omegaA = 3
# --- omgeaD = -3
# ---  N=12
# --- Coupling lambda = 10^(-3)
#The goal is to find the optimal values of chiA,chiD

    
class AverageProbability:

    def __init__(self,chiA,chiD,coupling_lambda,omegaA,omegaD,max_N,max_t):
        self.chiA,self.chiD = chiA,chiD
        self.coupling_lambda = coupling_lambda
        self.omegaA,self.omegaD = omegaA,omegaD
        self.max_N = max_N
        self.max_t = max_t


    def PDData(self):
       
        avgND = Execute(self.chiA,self.chiD,self.coupling_lambda,self.omegaA, self.omegaD,
                        self.max_N,self.max_t, data_dir="", return_data=True).executeOnce()

        t_span = range(0,self.max_t+1)
  

        avgPD= np.array(avgND)/ self.max_N
        #plt.plot(t_span,avgPD)
        #plt.show()
        avgPD = np.average(avgPD)
        return avgPD
        


class Reinforcement:


    def __init__(self,omegaD,omegaA,coupling_lambda,max_N,max_t,limitsxAxD,stepxA,stepxD,
                iterations,learning_rate):
        self.omegaD,self.omegaA = omegaD,omegaA
        self.coupling_lambda = coupling_lambda
        self.max_N = max_N
        self.max_t = max_t
        self.min_xA,self.max_xA,self.min_xD,self.max_xD = limitsxAxD
        self.stepxA,self.stepxD = stepxA,stepxD
        self.iterations = iterations
        self.learning_rate = learning_rate
        

    # ------- Step 1: Make an educated guess of the optimal values -------
    def InitialGuess(self):
        chiD_initial = (self.omegaA-self.omegaD)/self.max_N + np.random.normal(0,0.2,1)
        chiA_initial = -chiD_initial
        return chiA_initial,chiD_initial
    

    def change_xA(self,chiA_initial,chiD_initial):
        #2a) Optimal value of xA
        P_changexA = []
        P_changexA.append(1)
        count1,count2 = 0,0
        for iteration in range(self.iterations):
            chiAplus,chiAminus = chiA_initial + self.stepxA,chiA_initial - self.stepxA
            #Average Probability of being to donor
            avgPD_chiAplus = AverageProbability(chiA = chiAplus,chiD = chiD_initial,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            avgPD_chiAminus = AverageProbability(chiA = chiAminus,chiD = chiD_initial,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            P_avg = (avgPD_chiAplus + avgPD_chiAminus)/2
            P_changexA.append(P_avg)
            if count1 <= 5:
                if P_changexA[iteration+1]-P_changexA[iteration] < 10**(-2) and abs(P_avg-0.5) < 0.1: count1 +=1
                if count1 == 5: self.learning_rate = self.coupling_lambda/100
            if self.learning_rate == self.coupling_lambda/100 and count2 <= 5:
                if P_changexA[iteration+1]-P_changexA[iteration] < 10**(-3) and abs(P_avg-0.5) < 0.1: count2 +=1
                if count2 == 5: return chiA_initial
            print(r"Iteration {},C1={},C2={},Pavg = {},Learning rate = {}".format(iteration +1,count1,count2,P_avg,self.learning_rate) )
            der_xA = (0.5/self.stepxA)*(avgPD_chiAplus-avgPD_chiAminus)
            chiA_initial += -self.learning_rate*der_xA


    def change_xD(self,chiA_final,chiD_initial):
        P_changexD = []
        P_changexD.append(1)
        count1,count2 = 0,0
        for iteration in range(self.iterations):
            chiDplus,chiDminus = chiD_initial + self.stepxD,chiD_initial - self.stepxD
            #Average Probability of being to donor
            avgPD_chiDplus = AverageProbability(chiA = chiA_final,chiD = chiDplus,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            avgPD_chiDminus = AverageProbability(chiA = chiA_final,chiD = chiDminus,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            P_avg = (avgPD_chiDplus + avgPD_chiDminus)/2
            P_changexD.append(P_avg)
            if count1 <= 5:
                if P_changexD[iteration+1]-P_changexD[iteration] < 10**(-2) and abs(P_avg-0.5) < 0.02: count1 +=1
                if count1 == 5: self.learning_rate = self.coupling_lambda/100
            if self.learning_rate == self.coupling_lambda/100 and count2 <= 5:
                if P_changexD[iteration+1]-P_changexD[iteration] < 10**(-3) and abs(P_avg-0.5) < 0.02: count2 +=1
                if count2 == 5: return chiD_initial
            print(r"Iteration {},C1={},C2={},Pavg = {},Learning rate = {}".format(iteration +1,count1,count2,P_avg,self.learning_rate) )
            der_xD = (0.5/self.stepxD)*(avgPD_chiDplus-avgPD_chiDminus)
            chiD_initial += -self.learning_rate*der_xD

    # ------- Step 2: RL ------- 
    def train(self):
        chiA_initial,chiD_initial = self.InitialGuess()
        #2a) Change chiA
        chiA_final = self.change_xA(chiA_initial,chiD_initial)
        #2b) Change chiD
        #Reset the learning rate
        if self.coupling_lambda < 10**(-2): self.learning_rate = 1000*self.coupling_lambda
        else: self.learning_rate = 0.1*self.coupling_lambda
        chiD_final = self.change_xD(chiA_final,chiD_initial)

        print('Final: chiA,chiD = {},{}'.format(chiA_final,chiD_final))
        final_data = Execute(chiA_final,chiD_final,self.coupling_lambda, self.omegaA, self.omegaD, self.max_N, 
                            self.max_t, data_dir = "", return_data=True).executeOnce()
        t_span = range(0,self.max_t+1)
        plt.plot(t_span,final_data)
        plt.show()

        
            for iteration in range(self.iterations):
            #New values of chiA,chiD
            chiAplus,chiAminus = chiA_initial + self.stepxA,chiA_initial - self.stepxA
            chiDplus,chiDminus = chiD_initial + self.stepxD,chiD_initial - self.stepxD
            #Average Probability of being to donor
            avgPD_chiAplus = AverageProbability(chiA = chiAplus,chiD = chiD_initial,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            
            avgPD_chiAminus = AverageProbability(chiA = chiAminus,chiD = chiD_initial,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            avgPD_chiDplus = AverageProbability(chiA = chiA_initial,chiD = chiDplus,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            avgPD_chiDminus = AverageProbability(chiA = chiA_initial,chiD = chiDminus,
                                                coupling_lambda=self.coupling_lambda,
                                                omegaA= self.omegaA,
                                                omegaD= self.omegaD,
                                                max_N= self.max_N,
                                                max_t = self.max_t
                                                ).PDData()
            
            der_xD = (0.5/self.stepxD)*(avgPD_chiDplus-avgPD_chiDminus)
            der_xA = (0.5/self.stepxA)*(avgPD_chiAplus-avgPD_chiAminus)
            print(r"PD = {},(x_A,x_D) = ({},{})".format(avgPD_chiAplus,chiA_initial,chiD_initial))
            chiA_initial += -self.learning_rate*der_xA
            chiD_initial += -self.learning_rate*der_xD

if __name__=='__main__':
    coupling_parameter = 1
    if coupling_parameter <= 10**(-1): learning_rate = 1000*coupling_parameter
    else: learning_rate = 0.1*coupling_parameter
    problem = Reinforcement(omegaD=-3,omegaA=3,
                            coupling_lambda=coupling_parameter,
                            max_N=12,
                            max_t = 10**4,
                            limitsxAxD = [-2,2,-2,2],
                            stepxA=10**(-2),
                            stepxD =10**(-2),
                            iterations = 1000,
                            learning_rate = learning_rate)
    problem.train()
 """
