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

        """
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
        """

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
