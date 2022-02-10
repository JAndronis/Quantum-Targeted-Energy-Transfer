from calendar import SATURDAY
from debugpy import trace_this_thread
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from itertools import product
from Execute import Execute
from matplotlib import pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer
from tensorflow.keras.optimizers import SGD, Adam


def find_nearest_2D(array, value):
  valuex,valuey = value
  x_indices,y_indices = [],[]
  for i in range(len(array)):
    if abs(array[i][0] -valuex) < 10**(-9):
      x_indices.append(i)
    if abs(array[i][1] -valuey) <10**(-9):
      y_indices.append(i)

  return np.intersect1d(x_indices,y_indices)[0]



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



class Env:

  def __init__(self,States,PossibleActions,paramsxAxD,NpointsChiA,NpointsChiD):
    
    self.States = States
    self.PossibleActions = PossibleActions
    self.minxA,self.maxxA,self.minxD,self.maxxD = paramsxAxD
    self.NpointsChiA,self.NpointsChiD = NpointsChiA,NpointsChiD
    
    chiAs,chiDs = np.linspace(self.minxA,self.maxxA,self.NpointsChiA),np.linspace(self.minxD,self.maxxD,self.NpointsChiD)

    self.NStates,self.Nactions = len(self.States),len(list(self.PossibleActions.values()))
    self.stepxA,self.stepxD = np.diff(chiAs)[0],np.diff(chiDs)[0]
    self.Denied_minxA = [self.States[i] for i in range(0,self.NpointsChiD) ]
    self.Denied_maxxA = [self.States[i] for i in range(self.NStates-self.NpointsChiD,self.NStates) ]
    self.Denied_minxD = [self.States[i] for i in range(0,self.NStates,self.NpointsChiD) ]
    self.Denied_maxxD = [self.States[i] for i in range(self.NpointsChiD-1,self.NStates,self.NpointsChiD) ]



  def Step(self,action,CurrentChiA,CurrentChiD,coupling_lambda,omegaA,omegaD,maxN,maxt):
    # --------- Apply the action ---------
    #Fix Boundary
    if (CurrentChiA,CurrentChiD) in self.Denied_minxA + self.Denied_maxxA + self.Denied_minxD + self.Denied_maxxD:
      NewChiA,NewChiD = CurrentChiA,CurrentChiD
      print('edge')
    #Examine which action
    elif action == 0: 
      NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD
      #print('Enter 0 case')
    elif action == 1: 
      NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD
      #print('Enter 1 case')
    elif action == 2: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD - self.stepxD
      #print('Enter 2 case')
    elif action == 3: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD + self.stepxD
      #print('Enter 3 case')
    elif action == 4:
      NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD - self.stepxD
      #print('Enter 4 case')
    elif action == 5:
      NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD + self.stepxD
      #print('Enter 5 case')
    elif action == 6: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD
      #print('Enter 6 case')

    #print('Old (xA,xD) = {},{}, New (xA,xD) = {},{}'.format( CurrentChiA,CurrentChiD,NewChiA,NewChiD ) ) 
    # --------- New state ---------
    print("Current: ", CurrentChiA, CurrentChiD)
    print("New: ", NewChiA, NewChiD)
    NewStateIndex = find_nearest_2D(self.States,(NewChiA,NewChiD))

    # --------- Reward ---------
    NewStateAverageProbabilityCase = AverageProbability(chiA = NewChiA,chiD = NewChiD,
                                                        coupling_lambda = coupling_lambda,
                                                        omegaA = omegaA,
                                                        omegaD = omegaD,
                                                        max_N = maxN,
                                                        max_t = maxt)

    Reward = maxN*(1- 5*(NewStateAverageProbabilityCase.PDData()-0.5)**2 )

    # --------- Extra Info ---------
    Info = NewStateAverageProbabilityCase.PDData()

    print(30*'-')
    # --------- Done ---------
    if abs(Info-1) < 10**(-3) or abs(Info - 0.5) < 10**(-2): Done = True 
    else: Done = False

    return NewStateIndex,Reward,Done,Info


  """ 
  def KeepPath(self,Q_table,max_t):
    InitialStateIndex = np.random.choice(self.NStates)
    InitialState = self.States[InitialStateIndex]
    chiAInitial,chiDInitial = InitialState
    AvgProbInitialInstance = AverageProbability(chiAInitial,chiDInitial,self.coupling_lambda,
                                                self.omegaA,self.omegaD,self.max_N,max_t)
    PDInitial = AvgProbInitialInstance.PDData()
    Path = []
    Path.append(InitialStateIndex) 
    while abs(PDInitial - 0.5) > 0.01:
      CurrentStateIndex = InitialStateIndex
      CurrentChiA,CurrentChiD = self.States[CurrentStateIndex]
      action = np.argmax(Q_table[CurrentStateIndex,])
      NewStateIndex,_,_,Info = self.ApplyAction(action,CurrentChiA,CurrentChiD,max_t)
      Path.append(NewStateIndex)
      PDInitial = Info
      InitialStateIndex = NewStateIndex
    
    FinalChiA,FinalChiD = CurrentChiA,CurrentChiD
    print(FinalChiA,FinalChiD)

    FinalData = AverageProbability(FinalChiA,FinalChiD,self.coupling_lambda,self.omegaA,
                                                  self.omegaD,self.max_N,max_t)

    print('------------ This is the path followed ------------')
    for k in Path:
      print(self.States[k])

    T_span = np.arange(0,max_t+1)
    plt.plot(T_span,FinalData)
    plt.show()
 
"""
class Agent:

  def __init__(self,paramsxAxD,NpointsChiA,NpointsChiD,coupling_lambda,omegaA,omegaD,maxN,maxt):

    #Environment Parameters
    self.paramsxAxD = paramsxAxD
    self.minxA,self.maxxA,self.minxD,self.maxxD =self.paramsxAxD
    self.NpointsChiA,self.NpointsChiD = NpointsChiA,NpointsChiD
    self.coupling_lambda = coupling_lambda
    self.omegaA,self.omegaD = omegaA,omegaD
    self.max_N = maxN
    self.max_t = maxt

    #Possible Actions
    self.PossibleActions = {'Decrease xA':0,'Increase xA':1,
                            'Decrease xD':2,'Increase xD':3,
                            'Decrease both':4,'Increase both':5,
                            'Idle':6 }
    #Possible States
    chiAs,chiDs = np.linspace(self.minxA,self.maxxA,self.NpointsChiA),np.linspace(self.minxD,self.maxxD,self.NpointsChiD)
    self.States = list(product(chiAs,chiDs))
    self.NStates,self.Nactions = len(self.States),len(list(self.PossibleActions.values()))


  def CreateModel(self,InputShape):
    model = Sequential()
    model.add(InputLayer(input_shape=InputShape))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(self.Nactions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model

  def Train(self,EpsilonInitial,EpsilonDecay,Gamma,Episodes):
    Env_case = Env(self.States,self.PossibleActions,self.paramsxAxD,self.NpointsChiA,self.NpointsChiD)
    OneHotStates = np.identity(self.NStates)

    model = self.CreateModel(OneHotStates[0].shape)
    StateResetIndex = np.random.randint(0,self.NStates+1)

    for episode in range(Episodes):
      print('Episode = {} out of {}'.format(episode+1,Episodes))
      StateIndex = StateResetIndex
      Epsilon = EpsilonInitial
      Done = False
      #print(OneHotStates[StateIndex].shape == OneHotStates[0].shape)
      #print(OneHotStates[StateIndex].shape)
      #print(OneHotStates[0].shape)
    
      while not Done:
        Epsilon *= EpsilonDecay
        if np.random.uniform(0,1) < Epsilon:
            action = np.random.randint(0,self.Nactions)
        else:
          action = np.argmax(model.predict(np.expand_dims(OneHotStates[StateIndex],0)))
        
        (xAState,xDState) = self.States[StateIndex]

        NewStateIndex,Reward,Done,Info = Env_case.Step(action=action,
                                                      CurrentChiA = xAState,
                                                      CurrentChiD= xDState,
                                                      coupling_lambda = self.coupling_lambda,
                                                      omegaA = self.omegaA,
                                                      omegaD = self.omegaD,
                                                      maxN = self.max_N,
                                                      maxt = self.max_t)
        if Done: break

        target_vector = model.predict(np.expand_dims(OneHotStates[StateIndex],0))[0]
      
        target = Reward + Gamma * np.max(model.predict(np.expand_dims(OneHotStates[NewStateIndex],0)))
        
        target_vector[action] = target
        model.fit(np.expand_dims(OneHotStates[StateIndex],0),target_vector.reshape(-1, self.Nactions), epochs=1, verbose=0)
        if(StateIndex==NewStateIndex):
          StateIndex = np.random.randint(0,self.NStates+1)
        else:
          StateIndex = NewStateIndex
      


epsilon = 0.8
epsilon_decay = 0.95
gamma = 0.6
learning_rate = 0.6
episodes = 3

case_Agent = Agent(paramsxAxD=[-2,2,-2,2],NpointsChiA = 8,NpointsChiD=10,
                  coupling_lambda = 10**(-1),omegaA = 3,omegaD = -3,
                  maxN=12,
                  maxt= 10**4)

case_Agent.Train(EpsilonInitial = epsilon,
                EpsilonDecay = epsilon_decay,
                Gamma = gamma,
                Episodes = episodes)
