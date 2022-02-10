import numpy as np
from itertools import product
from tet.Execute import Execute
from matplotlib import pyplot as plt

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
    self.Denied_minxA = [self.States[i] for i in range(0,self.N_points) ]
    self.Denied_maxxA = [self.States[i] for i in range(self.NStates-self.N_points,self.NStates) ]
    self.Denied_minxD = [self.States[i] for i in range(0,self.NStates,self.N_points) ]
    self.Denied_maxxD = [self.States[i] for i in range(self.N_points-1,self.NStates,self.N_points) ]
    
    
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




  def ApplyAction(self,action,CurrentChiA,CurrentChiD,max_t):
    #Define the New State
    NewChiA,NewChiD = CurrentChiA,CurrentChiD
    if action == 0 and (CurrentChiA,CurrentChiD) not in self.Denied_minxA : 
      NewChiA,NewChiD = CurrentChiA - self.stepxA,CurrentChiD
      #print('Enter 0 case')
    if action == 1 and (CurrentChiA,CurrentChiD) not in self.Denied_maxxA: 
      NewChiA,NewChiD = CurrentChiA + self.stepxA,CurrentChiD
      #print('Enter 1 case')
    if action == 2 and (CurrentChiA,CurrentChiD) not in self.Denied_minxD: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD - self.stepxD
      #print('Enter 2 case')
    if action == 3 and (CurrentChiA,CurrentChiD)  not in self.Denied_maxxA: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD + self.stepxD
      #print('Enter 3 case')
    if action == 4: 
      NewChiA,NewChiD = CurrentChiA,CurrentChiD
      #print('Enter 4 case')

    #print('Old (xA,xD) = {},{}, New (xA,xD) = {},{}'.format( CurrentChiA,CurrentChiD,NewChiA,NewChiD ) ) 

    NewStateIndex = find_nearest_2D(self.States,(NewChiA,NewChiD))

    #Define the Reward
    CurrentAverageProbabilityCase = AverageProbability(NewChiA,NewChiD,self.coupling_lambda,
                                                      self.omegaA,self.omegaD,self.max_N,max_t)
    #Reward = 1 - (CurrentAverageProbabilityCase.PDData()-0.5)**2
    Reward = self.max_N - (self.max_N*CurrentAverageProbabilityCase.PDData()-0.5*self.max_N)**2
    #Define Info
    Info = CurrentAverageProbabilityCase.PDData()

    #Decide if Done
    if Info == 1 or abs(Info - 0.5) < 10**(-2): Done = True 
    else: Done = False

    return NewStateIndex,Reward,Done,Info


  #---------------- Step 2: Fill the Q-table ----------------
  def TrainAgent(self,epsilon,epsilon_decay,iterations,learning_rate,gamma,max_t):
    Q_table = np.zeros(shape = (self.NStates,self.Nactions))
    InitialStateIndex = np.random.choice(self.NStates)
    InitialState = self.States[InitialStateIndex]
    
    for iteration in range(iterations):
      CurrentState,CurrentStateIndex= InitialState,InitialStateIndex
      CurrentChiA,CurrentChiD = InitialState
      print('Iteration = {},xA = {},xD = {}'.format(iteration+1,round(CurrentChiA,4),
                                                    round(CurrentChiD,4)))
      #Pick an action
      random_number = np.random.randint(low=0,high=1)
      if random_number < epsilon:
        action = np.random.choice(list(self.PossibleActions.values()) )
      else:
        action = np.argmax(Q_table[CurrentState, :])
      epsilon *= epsilon_decay

      #New state,Reward,Done,Info after applying the action selected
      NewStateIndex,Reward,Done,Info = self.ApplyAction(action,CurrentChiA,CurrentChiD,max_t)
      if Done: break
      NewState = self.States[NewStateIndex]
      #Update Q-tabble
      Q_table[CurrentStateIndex, action] += Reward + learning_rate * \
        (gamma * np.max(Q_table[NewStateIndex,:]) - Q_table[CurrentStateIndex, action])

      InitialState,InitialStateIndex = NewState,NewStateIndex

    #Print the path followed until the optimal values
    self.KeepPath(Q_table,max_t)
  



#---------------- Step 1a: Parameters of the problem ----------------
Agent = EducateAgent(paramsxAxD=[-2,2,-2,2,1000],coupling_lambda = 10**(-3),omegaA = 3,omegaD = -3,
                    maxN=12)

#---------------- Step 1b: Parameters of the agent's training ----------------
epsilon = 0.8
epsilon_decay = 0.95
gamma = 0.6
learning_rate = 0.6
iterations = 50

q, inf = Agent.TrainAgent(epsilon,epsilon_decay,iterations,learning_rate,gamma,max_t = 10**2)
print(q)
print(inf)
