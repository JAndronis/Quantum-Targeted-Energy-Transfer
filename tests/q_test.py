import numpy as np
from itertools import product
from data_process import writeData,LoadModel
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from RL_Q_learning_NN import Agent


NumberOfRuns = 8
for Run in range(1,NumberOfRuns+1):
    agent = Agent()
    agent.ExecutionUpdateTry = Run 
    if Run == 1: ModelExists = False
    else: ModelExists = True
    print('\n\nUpdate Try {} out of {}'.format(agent.ExecutionUpdateTry,NumberOfRuns))
    agent.TrainTotal(ModelExists = ModelExists,NumberOfRuns = NumberOfRuns)

