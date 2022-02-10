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
np.random.seed(57)

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
    
    avgPD = np.average(avgPD)
    return avgPD



class Env:

  def __init__(self,States,PossibleActions,paramsxAxD,NpointsChiA,NpointsChiD):
    
    self.States = States
    self.PossibleActions = PossibleActions
    self.minxA,self.maxxA,self.minxD,self.maxxD = paramsxAxD
    self.NpointsChiA,self.NpointsChiD = NpointsChiA,NpointsChiD
    
    self.chiAs,self.chiDs = np.linspace(self.minxA,self.maxxA,self.NpointsChiA),np.linspace(self.minxD,self.maxxD,self.NpointsChiD)

    self.NStates,self.Nactions = len(self.States),len(list(self.PossibleActions.values()))
    self.stepxA,self.stepxD = np.diff(self.chiAs)[0],np.diff(self.chiDs)[0]
    self.Denied_minxA = [self.States[i] for i in range(0,self.NpointsChiD) ]
    self.Denied_maxxA = [self.States[i] for i in range(self.NStates-self.NpointsChiD,self.NStates) ]
    self.Denied_minxD = [self.States[i] for i in range(0,self.NStates,self.NpointsChiD) ]
    self.Denied_maxxD = [self.States[i] for i in range(self.NpointsChiD-1,self.NStates,self.NpointsChiD) ]

  def _getReward(self, edge, CurrentChiA, CurrentChiD, NewChiA, NewChiD, coupling_lambda, omegaA, omegaD, maxN, maxt):
    # --------- New state --------
    # print("Current: ", round(CurrentChiA,5), round(CurrentChiD,5))
    print("New: ", round(NewChiA,5), round(NewChiD,5))
    NewStateIndex = find_nearest_2D(self.States,(NewChiA,NewChiD))

    # --------- Reward ---------
    NewStateAverageProbabilityCase = AverageProbability(chiA = NewChiA,chiD = NewChiD,
                                                        coupling_lambda = coupling_lambda,
                                                        omegaA = omegaA,
                                                        omegaD = omegaD,
                                                        max_N = maxN,
                                                        max_t = maxt)

    if edge:
      Reward = maxN*(1- 50*abs(NewStateAverageProbabilityCase.PDData()-0.5) )
    else:
      Reward = maxN*(1- 5*abs(NewStateAverageProbabilityCase.PDData()-0.5) )

    return Reward, NewStateIndex, NewStateAverageProbabilityCase

  def Step(self,action,CurrentChiA,CurrentChiD,coupling_lambda,omegaA,omegaD,maxN,maxt):
    # --------- Apply the action ---------
    #Fix Boundary. Demand to start a new episode
    if (CurrentChiA,CurrentChiD) in self.Denied_minxA + self.Denied_maxxA + self.Denied_minxD + self.Denied_maxxD:
      # randomchoice = 4*np.random.sample(2)-2  # random choice from [-2,2)
      # print(randomchoice)
      NewChiA, NewChiD = np.random.choice(self.chiAs), np.random.choice(self.chiDs)
      print('edge')
      edge = True
      # Done = True
      # return None,None,Done,None
    
    else:
      edge = False
      match action:
        case 0:
            NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD
        case 1:
            NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD
        case 2:
            NewChiA,NewChiD = CurrentChiA,CurrentChiD - self.stepxD
        case 3:
            NewChiA,NewChiD = CurrentChiA,CurrentChiD + self.stepxD
        case 4:
            NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD - self.stepxD
        case 5:
            NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD + self.stepxD
        case 6:
            NewChiA,NewChiD = CurrentChiA,CurrentChiD
        case _:
            return "Non valid action"


    #print('Old (xA,xD) = {},{}, New (xA,xD) = {},{}'.format( CurrentChiA,CurrentChiD,NewChiA,NewChiD ) ) 
    # # --------- New state --------
    # print("Current: ", round(CurrentChiA,5), round(CurrentChiD,5))
    # print("New: ", round(NewChiA,5), round(NewChiD,5))
    # NewStateIndex = find_nearest_2D(self.States,(NewChiA,NewChiD))

    # # --------- Reward ---------
    # NewStateAverageProbabilityCase = AverageProbability(chiA = NewChiA,chiD = NewChiD,
    #                                                     coupling_lambda = coupling_lambda,
    #                                                     omegaA = omegaA,
    #                                                     omegaD = omegaD,
    #                                                     max_N = maxN,
    #                                                     max_t = maxt)

    # Reward = maxN*(1- 5*abs(NewStateAverageProbabilityCase.PDData()-0.5) )

    Reward, NewStateIndex, NewStateAverageProbabilityCase = self._getReward(edge,
                                                                            CurrentChiA, 
                                                                            CurrentChiD, 
                                                                            NewChiA, 
                                                                            NewChiD, 
                                                                            coupling_lambda,
                                                                            omegaA, 
                                                                            omegaD, 
                                                                            maxN, 
                                                                            maxt)

    # --------- Extra Info ---------
    Info = NewStateAverageProbabilityCase.PDData()
    print('Info:  {}  '.format(round(Info,5)))

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
    StateResetIndex = np.random.randint(0,self.NStates)
    RewardsList = []

    # for now
    max_iter = 5
    counter = 0

    for episode in range(Episodes):

      print('-'*15 + f'> Episode = {episode+1} out of {Episodes} <' + '-'*15)
      StateIndex = StateResetIndex
      Epsilon = EpsilonInitial
      Done = False
      SumReward = 0
      CounterSum = 0

      counter = 0

      while not Done:
        counter += 1
        if counter >= max_iter: Done = True
        else:
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
        
        if Done and CounterSum == 0:
          print('Unfortunate initial guess,try again')
          exit()
        elif Done:
          RewardsList.append(SumReward/CounterSum)
          break
        
        SumReward += Reward
        target_vector = model.predict(np.expand_dims(OneHotStates[StateIndex],0))[0]
      
        target = Reward + Gamma * np.max(model.predict(np.expand_dims(OneHotStates[NewStateIndex],0)))
        
        target_vector[action] = target
        model.fit(np.expand_dims(OneHotStates[StateIndex],0),target_vector.reshape(-1, self.Nactions), epochs=1, verbose=0)
        StateIndex = NewStateIndex

        CounterSum +=1

    plt.figure()
    plt.plot(np.arange(1,Episodes+1), RewardsList)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()

epsilon = 0.8
epsilon_decay = 0.95
gamma = 0.6
learning_rate = 0.6
episodes = 2

case_Agent = Agent(paramsxAxD=[-2,2,-2,2],NpointsChiA = 5,NpointsChiD=7,
                  coupling_lambda = 10**(-1),omegaA = 3,omegaD = -3,
                  maxN=12,
                  maxt= 10**4)

case_Agent.Train(EpsilonInitial = epsilon,
                EpsilonDecay = epsilon_decay,
                Gamma = gamma,
                Episodes = episodes)
