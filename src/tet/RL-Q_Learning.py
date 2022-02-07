import numpy as np
from itertools import product
from Execute import Execute

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx


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
        #Period= ExtractPeriod(xdata=t_span,ydata = avgPD,maxt=self.max_t).EstimatePeriod()

        #avgPD = np.average(avgPD[0:self.find_nearest(t_span,2*Period)] )
        avgPD = np.average(avgPD)
        return avgPD


class EducateAgent:

    def __init__(self,paramsxAxD,coupling_lambda,omegaA,omegaD,maxN):
      self.minxA,self.maxxA,self.minxD,self.maxxD,self.N_points =paramsxAxD
      self.coupling_lambda = coupling_lambda
      self.omegaA,self.omegaD = omegaA,omegaD
      self.max_N = maxN
      #Possible Actions
      self.PossibleActions = {'Decrease xA':0,'Increase xA':1,
                              'Decrease xD':2,'Increase xD':3,
                              'Idle':4}
      #Possible States
      chiAs,chiDs = np.linspace(self.minxA,self.maxxA,self.N_points),np.linspace(self.minxD,self.maxxD,self.N_points)
      self.States = list(product(chiAs,chiDs))
      self.NStates,self.Nactions = len(self.States),len(list(self.PossibleActions.values()))
      self.stepxA,self.stepxD = np.diff(chiAs)[0],np.diff(chiDs)[0]
    
    
    def ApplyAction(self,action,CurrentChiA,CurrentChiD,max_t):
      #Define the New State
      if action == 0: NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD
      if action == 1: NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD
      if action == 2: NewChiA,NewChiD = CurrentChiA,CurrentChiD - self.stepxD
      if action == 3: NewChiA,NewChiD = CurrentChiA,CurrentChiD + self.stepxD
      if action == 4: NewChiA,NewChiD = CurrentChiA,CurrentChiD

      NewStateIndex = find_nearest(self.States,(NewChiA,NewChiD))

      #Define the Reward
      CurrentAverageProbabilityCase = AverageProbability(NewChiA,NewChiD,self.coupling_lambda,
                                                            self.omegaA,self.omegaD,
                                                            self.max_N,max_t)
      Reward = 1 - (CurrentAverageProbabilityCase.PDData()-0.5)**2

      #Define Info
      Info = CurrentAverageProbabilityCase.PDData()

      #Decide if Done
      if Info == 1 or abs(Info - 0.5) < 10**(-3): Done = True 
      else: Done = False

      return NewStateIndex,Reward,Done,Info



    def TrainAgent(self,epsilon,epsilon_decay,iterations,learning_rate,gamma,max_t):
      Q_table = np.zeros(shape = (self.NStates,self.Nactions))
      InitialStateIndex = np.random.choice(self.NStates)
      InitialState = self.States[InitialStateIndex]
      Infos = []
    
      for _ in range(iterations):
        print(_)
        CurrentState,CurrentStateIndex= InitialState,InitialStateIndex
        CurrentChiA,CurrentChiD = InitialState
        #Pick an action
        random_number = np.random.randint(low=0,high=1)
        if random_number < epsilon:
          #Pick randomly an action
          action = np.random.choice(list(self.PossibleActions.values()) )
        else:
          action = np.argmax(Q_table[CurrentState, :])
        epsilon *= 1/epsilon_decay

        NewStateIndex,Reward,Done,Info = self.ApplyAction(action,CurrentChiA,CurrentChiD,max_t)
        if Done: break
        Infos.append(Info)
        NewState = self.States[NewStateIndex]

        Q_table[CurrentStateIndex, action] += Reward + learning_rate * \
          (gamma * np.max(Q_table[NewStateIndex,:]) - Q_table[CurrentStateIndex, action])

        CurrentState = NewState

      return Q_table,Infos



Agent = EducateAgent(paramsxAxD=[-2,2,-2,2,1500],coupling_lambda = 10**(-1),omegaA = 3,omegaD = -3,
                    maxN=12)

epsilon = 0.8
epsilon_decay = 0.95
gamma = 0.6
learning_rate = 0.6
iterations = 10


Agent.TrainAgent(epsilon,epsilon_decay,iterations,learning_rate,gamma,max_t = 10**4)

