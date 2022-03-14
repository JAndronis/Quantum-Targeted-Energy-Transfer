from ast import Load
import numpy as np
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from calendar import SATURDAY
from debugpy import trace_this_thread
from itertools import product
from Execute import Execute
from saveFig import saveFig
from matplotlib import pyplot as plt
from data_process import createDir,SaveWeights,LoadModel,writeData,read_1D_data,ReadDeque
from itertools import chain
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,InputLayer
from tensorflow.keras.optimizers import SGD, Adam


np.random.seed(57)


# ------------- First Set of Parameters ------------- 
minxA,maxxA,minxD,maxxD = -2,2,-2,2
data_dir = f"{os.getcwd()}/data"
createDir(data_dir,replace = True)

ParametersFirstSet = {'ParamschiAchiD': [minxA,maxxA,minxD,maxxD],'NpointsChiA':100,
                      'NpointsChiD':100,'coupling_lambda':0.1,
                      'maxN':12,'maxt':10**2,'data_dir': data_dir,
                      'omegaA': 3,'omegaD':-3  }

ChiAs = np.linspace(minxA,maxxA,ParametersFirstSet['NpointsChiA'])
ChiDs = np.linspace(minxD,maxxD,ParametersFirstSet['NpointsChiD'])

ParametersFirstSet = {**ParametersFirstSet, **{'ChiAs':ChiAs},**{'ChiDs':ChiDs} }

#ParametersFirstSet = ParametersFirstSet | {'ChiAs':ChiAs} | {'ChiDs':ChiDs}
# ------------- Second Set of Parameters ------------- 
ParametersSecondSet = {'EpsilonInitial': 1,'EpsilonDecay':0.995,'MinEpsilon':10**(-3),
                        'Gamma':0.90,'Episodes':250, 'MaxIterations':50,
                        'ReplayMemorySize': 1000,'MinReplayMemorySize': 128, 
                        'BatchSize': 32,'UpdateWeigthsTargetCounter':64,
                        'ShowEvery':10}

# ------------- Total Parameters -------------
TotalParameters = {**ParametersFirstSet, **ParametersSecondSet }
#TotalParameters = ParametersFirstSet | ParametersSecondSet

#------------- Dictionary with actions -------------
PossibleActions = { 'Decrease xA':0,'Increase xA':1,
                    'Decrease xD':2,'Increase xD':3,
                    'Decrease both':4,'Increase both':5,
                    'Idle':6 }


class AverageProbability:

  def __init__(self,CaseChiA,CaseChiD):
    self.CaseChiA,self.CaseChiD = CaseChiA,CaseChiD
    self.coupling_lambda = TotalParameters['coupling_lambda']
    self.omegaA,self.omegaD = TotalParameters['omegaA'],TotalParameters['omegaD']
    self.maxN = TotalParameters['maxN']
    self.maxt = TotalParameters['maxt']
    self.datadir = TotalParameters['data_dir']

  def PDData(self):
       
    avgND = Execute(chiA = self.CaseChiA,chiD = self.CaseChiD,
                    coupling_lambda = self.coupling_lambda,
                    omegaA = self.omegaA, omegaD = self.omegaD, max_N= self.maxN,
                    max_t = self.maxt, data_dir=self.datadir, return_data=True).executeOnce()

    #t_span = range(0,self.maxt+1)

    avgPD= np.array(avgND)/ self.maxN
    
    avgPD_OverTime = np.average(avgPD)
    return avgPD_OverTime


class Env:

  def __init__(self):
    
    self.NpointsChiA,self.NpointsChiD = TotalParameters['NpointsChiA'],TotalParameters['NpointsChiD']   
    self.PossibleActions = PossibleActions
    self.ChiAs,self.ChiDs = TotalParameters['ChiAs'],TotalParameters['ChiDs']
    self.States = list(product(self.ChiAs,self.ChiDs))
    self.NStates,self.Nactions = len(self.States),len(list(self.PossibleActions.values()))
    self.stepChiA,self.stepChiD = np.diff(self.ChiAs)[0],np.diff(self.ChiDs)[0]
    self.coupling_lambda =TotalParameters['coupling_lambda']
    self.omegaA,self.omegaD = TotalParameters['omegaA'],TotalParameters['omegaD']
    self.maxN,self.maxt = TotalParameters['maxN'],TotalParameters['maxt']

    self.Denied_minxA = [i for i in range(0,self.NpointsChiD) ]
    self.Denied_maxxA = [i for i in range(self.NStates-self.NpointsChiD,self.NStates) ]
    self.Denied_minxD = [i for i in range(0,self.NStates,self.NpointsChiD) ]
    self.Denied_maxxD = [i for i in range(self.NpointsChiD-1,self.NStates,self.NpointsChiD) ]
    self.BoundaryLimitsIndices = self.Denied_minxA + self.Denied_maxxA + self.Denied_minxD + self.Denied_maxxD

  def FindNearest2D(self,array, value):
    valuex,valuey = value
    x_indices,y_indices = [],[]
    for i in range(len(array)):
      if abs(array[i][0] -valuex) < 10**(-9): x_indices.append(i)
      if abs(array[i][1] -valuey) <10**(-9): y_indices.append(i)
      
    return np.intersect1d(x_indices,y_indices)[0]


  def GetReward(self, Edge, NewChiA, NewChiD):
    # --------- Find NewStateIndex given NewChiA,NewChiD --------
    NewStateIndex = self.FindNearest2D(self.States,(NewChiA,NewChiD))

    # --------- Reward ---------
    NewStateAverageProbabilityData = AverageProbability(CaseChiA = NewChiA,CaseChiD= NewChiD).PDData()

    if Edge: Reward = self.maxN*(1- 50*abs(NewStateAverageProbabilityData-0.5) )
    else:Reward = self.maxN*(1- 5*abs(NewStateAverageProbabilityData-0.5) )

    return Reward, NewStateIndex, NewStateAverageProbabilityData



  def Step(self,Action,CurrentStateIndex):
    # Check Boundary. Reset(No new episode) given the respective penalnty 
    if CurrentStateIndex in self.BoundaryLimitsIndices: 
      NewStateIndex = np.random.randint(self.NStates)
      NewChiA, NewChiD = self.States[NewStateIndex]
      #print('edge')
      Edge = True

    else:
      Edge = False
      CurrentChiA,CurrentChiD = self.States[CurrentStateIndex]
      # match action:
      if Action == 0:NewChiA,NewChiD = CurrentChiA - self.stepChiA,CurrentChiD

      elif Action == 1:NewChiA,NewChiD = CurrentChiA + self.stepChiA,CurrentChiD

      elif Action == 2:NewChiA,NewChiD = CurrentChiA,CurrentChiD - self.stepChiD
          
      elif Action == 3:NewChiA,NewChiD = CurrentChiA,CurrentChiD + self.stepChiD
          
      elif Action == 4:NewChiA,NewChiD = CurrentChiA - self.stepChiA,CurrentChiD - self.stepChiD
          
      elif Action == 5:NewChiA,NewChiD = CurrentChiA + self.stepChiA,CurrentChiD + self.stepChiD
          
      elif Action == 6:NewChiA,NewChiD = CurrentChiA,CurrentChiD
          
      else:return "Non valid action"
          

    Reward, NewStateIndex, Info = self.GetReward(Edge, NewChiA, NewChiD) 
                                                                            
    # --------- Done ---------
    if abs(Info-1) < 10**(-3) or abs(Info - 0.5) < 10**(-2): Done = True 
    else: Done = False

    return NewStateIndex,Reward,Done,Info


class Agent: 

  def __init__(self):

    self.Episodes = TotalParameters['Episodes']
    self.Gamma = TotalParameters['Gamma']
    self.EpsilonDecay = TotalParameters['EpsilonDecay']
    self.MaxIterations = TotalParameters['MaxIterations']
    self.ReplayMemorySize = TotalParameters['ReplayMemorySize']
    self.MinReplayMemorySize = TotalParameters['MinReplayMemorySize']
    self.BatchSize = TotalParameters['BatchSize']
    self.Nactions = len(list(PossibleActions.values()))
    self.Epsilon,self.MinEpsilon =TotalParameters['EpsilonInitial'],TotalParameters['MinEpsilon']
    self.datadir = TotalParameters['data_dir']
    

    self.ReplayMemoryDeque = deque(maxlen=self.ReplayMemorySize)
    self.OneHotStates = np.identity(len(list(product(TotalParameters['ChiAs'],TotalParameters['ChiDs']))))
    self.model,self.TargetModel = self.CreateModel(),self.CreateModel()
    self.TargetModel.set_weights(self.model.get_weights())
    self.UpdateWeightsTarget,self.ExecutionUpdateTry  = 0,0


  def PlotResults(self):
    
    FinalDataEpsilons = read_1D_data(self.datadir,name_of_file='Epsilons.txt')
    FinalDataRewards = read_1D_data(self.datadir,name_of_file='Rewards.txt')
    FinalDataAvgProbs =  read_1D_data(self.datadir,name_of_file='AverageProbs.txt')

    plt.figure(figsize = (12,12))

    plt.subplot(2,2,1)
    plt.plot(np.arange(1,self.Episodes*self.ExecutionUpdateTry +1), FinalDataEpsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.subplot(2,2,2)
    plt.plot(np.arange(1,self.Episodes*self.ExecutionUpdateTry +1), FinalDataRewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(2,2,3)
    plt.plot(np.arange(1,self.Episodes*self.ExecutionUpdateTry +1), FinalDataAvgProbs)
    plt.xlabel('Episode')
    plt.ylabel(r'<P$_{D}$>_${t,iterations}$')


    saveFig('FinalData', self.datadir)
    plt.show()

    #os.remove(self.datadir)


  def SaveData(self):
    #Save rewards list
    writeData(data=self.RewardsList, destination=self.datadir, name_of_file='Rewards.txt')
    #Save AverageProbs list
    writeData(data=self.AverageProbs, destination=self.datadir, name_of_file='AverageProbs.txt')
    #Save epsilons list
    writeData(data=self.Epsilons, destination=self.datadir, name_of_file='Epsilons.txt')
    #Save the current ReplayMemoryDeque
    writeData(data=self.ReplayMemoryDeque, destination=self.datadir, name_of_file='Deque.txt')
    #Save the value of self.UpdateWeightsTarget
    writeData(data = [self.UpdateWeightsTarget],destination=self.datadir,name_of_file='UpdateTarget.txt')
    #Save the value of self.ExecutionUpdateTry 
    #writeData(data = [self.ExecutionUpdateTry],destination=self.datadir,name_of_file='Try.txt')
    #Save current weigths from self.model
    SaveWeights(ModelToSave=self.model,case = 'model',destination=self.datadir)
    #Save current weights from self.TargetModel
    SaveWeights(ModelToSave=self.TargetModel,case = 'TargetModel',destination=self.datadir)
  

  def CreateModel(self):
    model = Sequential()
    model.add(InputLayer(input_shape=self.OneHotStates[0].shape))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(self.Nactions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model


  def DeriveBatch(self):
    #Structure of Batch: (CurrentStateIndex, Action, Reward, NewStateIndex, Done)
    if len(self.ReplayMemoryDeque) < self.MinReplayMemorySize: return 0
    else: return random.sample(self.ReplayMemoryDeque, self.BatchSize)


  def TrainEpisode(self, Done):
    
    #Decide if Batch will be derived
    if self.DeriveBatch() == 0:return
    else: Batch = self.DeriveBatch()

    CurrentStatesIndices = [case[0] for case in Batch]
    CurrentQsList = [self.model.predict(np.expand_dims(self.OneHotStates[index],0) for index in CurrentStatesIndices) ][0]
 
    NewStatesIndices = [case[3] for case in Batch]
    NewQsList = [self.TargetModel.predict(np.expand_dims(self.OneHotStates[index],0) for index in NewStatesIndices) ][0]
    

    StatesFit,QValuesFit = [],[]

    
    for index, (CurrentStateIndex, Action, Reward, NewStateIndex, Done) in enumerate(Batch):
     
      if not Done: NewQIndex = Reward + self.Gamma*np.max(NewQsList[index])
      else: NewQIndex = Reward

      self.CumulativeReward += Reward

      # Update Q value for given state
      CurrentQsList[index][Action] = NewQIndex
      # And append to our training data
      StatesFit.append(self.OneHotStates[CurrentStateIndex])
      QValuesFit.append(CurrentQsList[index])

    
    self.model.fit(np.array(StatesFit), np.array(QValuesFit).reshape(-1,self.Nactions), batch_size=self.BatchSize, verbose=0, shuffle=False)

    # Track of updating weights to target model
    if Done: self.UpdateWeightsTarget += 1

    # If counter reaches set value, update target network with weights of main network
    if self.UpdateWeightsTarget == TotalParameters['UpdateWeigthsTargetCounter']:
      #print('Updated')
      self.TargetModel.set_weights(self.model.get_weights())
      self.UpdateWeightsTarget = 0


 
  def TrainTotal(self,ModelExists,NumberOfRuns):

    if ModelExists:
      self.Epsilon = self.EpsilonDecay*read_1D_data(self.datadir,'Epsilons.txt')[(self.ExecutionUpdateTry-1)*self.Episodes-1]
      self.ReplayMemoryDeque = ReadDeque(self.datadir,'Deque.txt')
      self.model = LoadModel(case = 'model',destination=self.datadir)
      self.TargetModel = LoadModel(case = 'TargetModel',destination=self.datadir)
      self.UpdateWeightsTarget =int(read_1D_data(self.datadir,name_of_file='UpdateTarget.txt')[self.ExecutionUpdateTry-2])
    
    self.RewardsList,self.Epsilons,self.AverageProbs = [],[],[]
    
    for episode in range(self.Episodes):
      print('\rEpisode = {} out of {}'.format(episode+1,self.Episodes) ,end = "")
      #Begin a new episode
      self.CumulativeReward,self.AverageProb= 0,0
      IterationsCounter = 1
      
      #This is the reset state. The nomeclature is followed for iteration sake
      CurrentStateIndex = np.random.randint(len(self.OneHotStates))
      Done = False
      

      while not Done:
        
        if np.random.uniform() > self.Epsilon: Action = np.argmax(self.model.predict(np.expand_dims(
                                                        self.OneHotStates[CurrentStateIndex],0) ) )
        else: Action = np.random.randint(0,self.Nactions)
  
        NewStateIndex,Reward,Done,Info = Env().Step(Action,CurrentStateIndex)

        if IterationsCounter >= self.MaxIterations: Done = True
        #if IterationsCounter % TotalParameters['ShowEvery'] == 0:
          #print(f'Counter:{IterationsCounter}',end = "")
        #Update Replay Memory Deque
        self.ReplayMemoryDeque.append((CurrentStateIndex, Action, Reward, NewStateIndex, Done))
        
        self.TrainEpisode(Done)

        CurrentStateIndex = NewStateIndex
        IterationsCounter += 1
        self.AverageProb += Info

  
      self.RewardsList.append(self.CumulativeReward)
      self.AverageProbs.append(self.AverageProb/IterationsCounter)
      self.Epsilons.append(self.Epsilon)

      
      # Update epsilon value
      if self.Epsilon > self.MinEpsilon:
        self.Epsilon *= self.EpsilonDecay
        self.Epsilon = max(self.MinEpsilon, self.Epsilon)

    self.SaveData()
    if self.ExecutionUpdateTry == NumberOfRuns: self.PlotResults()

    
      

