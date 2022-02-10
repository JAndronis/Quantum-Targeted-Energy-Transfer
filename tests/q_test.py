from tet.RL_Q_learning_NN import Agent
from tet.data_process import createDir
import os

if __name__=="__main__":
	epsilon = 0.8
	epsilon_decay = 0.95
	gamma = 0.6
	learning_rate = 0.6
	episodes = 50

	cwd = os.getcwd()
	data = f"{cwd}/data"
	createDir(data)

	case_Agent = Agent(paramsxAxD=[-2,2,-2,2], 
					NpointsChiA=100, NpointsChiD=100,
					coupling_lambda = 10**(-1),
					omegaA = 3,omegaD = -3,
					maxN=12,
					maxt= 10**2,
					data_dir=data)

	case_Agent.Train(EpsilonInitial = epsilon,
					EpsilonDecay = epsilon_decay,
					Gamma = gamma,
					Episodes = episodes,
					max_iter=100)