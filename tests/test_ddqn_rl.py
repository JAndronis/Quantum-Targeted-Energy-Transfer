import numpy as np
import matplotlib.pyplot as plt
from tet.Execute import Execute

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

import itertools
import os
import gym

np.random.seed(42)
tf.random.set_seed(42)

MAX_BOSONS_DONOR = 12
MAX_BOSONS_ACCEPTOR = 12
CHI_MIN = -5
CHI_MAX = 5
COUPLING = 1
OMEGA_D = 3
OMEGA_A = -3
MAX_T = 100

class DuelingDeepQNetwork(keras.Model):
	def __init__(self, n_actions, fc1_dims, fc2_dims):
		super(DuelingDeepQNetwork, self).__init__()
		self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
		self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
		self.V = keras.layers.Dense(1, activation=None)
		self.A = keras.layers.Dense(n_actions, activation=None)

	def call(self, state):
		x = self.dense1(state)
		x = self.dense2(x)
		V = self.V(x)
		A = self.A(x)

		Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

		return Q

	def advantage(self, state):
		x = self.dense1(state)
		x = self.dense2(x)
		A = self.A(x)

		return A


class ReplayBuffer():
	def __init__(self, max_size, input_shape):
		self.mem_size = max_size
		self.mem_cntr = 0

		self.state_memory = np.zeros((self.mem_size, *input_shape),
										dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_shape),
										dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done

		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size, replace=False)

		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, new_states, dones

class Agent():
	def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
				 input_dims, epsilon_dec=1e-3, eps_end=0.01, 
				 mem_size=100000, fc1_dims=128,
				 fc2_dims=128, replace=100):
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = eps_end
		self.replace = replace
		self.batch_size = batch_size

		self.learn_step_counter = 0
		self.memory = ReplayBuffer(mem_size, input_dims)
		self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
		self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

		self.q_eval.compile(optimizer=Adam(learning_rate=lr),
							loss='mean_squared_error')
		# just a formality, won't optimize network
		self.q_next.compile(optimizer=Adam(learning_rate=lr),
							loss='mean_squared_error')

	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, observation):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			state = np.array([observation])
			actions = self.q_eval.advantage(state)
			action = tf.math.argmax(actions, axis=1).numpy()[0]

		return action

	def learn(self):
		if self.memory.mem_cntr < self.batch_size:
			return

		if self.learn_step_counter % self.replace == 0:
			self.q_next.set_weights(self.q_eval.get_weights())

		states, actions, rewards, states_, dones = \
									self.memory.sample_buffer(self.batch_size)

		q_pred = self.q_eval(states)
		q_next = self.q_next(states_)
		# changing q_pred doesn't matter because we are passing states to the train function anyway
		# also, no obvious way to copy tensors in tf2?
		q_target = q_pred.numpy()
		max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
		
		# improve on my solution!
		for idx, terminal in enumerate(dones):
			#if terminal:
				#q_next[idx] = 0.0
			q_target[idx, actions[idx]] = rewards[idx] + \
					self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
		self.q_eval.train_on_batch(states, q_target)

		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
						self.eps_min else self.eps_min

		self.learn_step_counter += 1

class CoupledBreathersEnv(gym.Env):
    def __init__(self):
        super(CoupledBreathersEnv, self).__init__()
        
        # s = [xD, xA]
        self.observation_space = gym.spaces.Box(low=np.array([0, CHI_MIN, CHI_MIN]), 
                                                high=np.array([MAX_BOSONS_ACCEPTOR, CHI_MAX, CHI_MAX]),
                                                shape=(3,),
                                                dtype=np.float32)
        
        self.temp_ar = []
        self._temp_ar = np.linspace(CHI_MIN, CHI_MAX, 101)
        for val in itertools.combinations(self._temp_ar, r=2):
            self.temp_ar.append(val)
            
        x = np.random.randint(len(self.temp_ar))
        
        self.chiA, self.chiD = self.temp_ar[x], self.temp_ar[x]
        
        # a = [delta xD, delta xA]
        self.action_space = gym.spaces.Discrete(len(self.temp_ar))
        
        self.reward = MAX_BOSONS_DONOR - MAX_BOSONS_DONOR
        
    def reset(self):
        x = np.random.randint(len(self.temp_ar))
        self.chiA, self.chiD = self.temp_ar[x][0], self.temp_ar[x][1]
        return np.array([MAX_BOSONS_DONOR, self.chiA, self.chiD], dtype=np.float32)
    
    def step(self, action):
        self.chiA, self.chiD = self.temp_ar[action][0], self.temp_ar[action][1]
        
        avgN = Execute(chiA=self.chiA,
                       chiD=self.chiD,
                       max_N=MAX_BOSONS_DONOR,
                       max_t=MAX_T,
                       coupling_lambda=COUPLING,
                       omegaA=OMEGA_A,
                       omegaD=OMEGA_D,
                       data_dir=os.getcwd(),
                       return_data=True)()
        
        minN = np.min(avgN)
        
        self.observation = np.array([minN, self.chiA, self.chiD], dtype=np.float32)
        
        self.reward = MAX_BOSONS_DONOR - minN
        done = bool(self.reward >= 11)
        
        info = {'Min Bosons on donor':minN}
        
        return self.observation, self.reward, done, info

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N): running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

if __name__ == "__main__":
	env = CoupledBreathersEnv()
	n_games = 100
	agent = Agent(lr=0.01, gamma=0.99, n_actions=env.action_space.n, epsilon=1.0, 
				batch_size=128, input_dims=[3], eps_end=0.01, epsilon_dec=1e-4)
	ddqn_scores = []
	eps_history = []

	for i in range(n_games):
		done = False
		score = 0
		observation = env.reset()
		steps = 0
		avg_info = 0
		while not done and steps<100:
			steps += 1
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.store_transition(observation, action, reward, observation_, done)
			observation = observation_
			agent.learn()
			avg_info += info['Min Bosons on donor']
			print('\rStep #', end='%d' % steps)
		avg_info /= steps
		eps_history.append(agent.epsilon)

		ddqn_scores.append(score)

		avg_score = np.mean(ddqn_scores[-100:])
		print('\nepisode: ', i,
			' epsilon: %.2f' % agent.epsilon,
			' Average min number of bosons on donor site: %.2f' % avg_info)

	filename = 'coupledsystem-dueling_ddqn.png'

	x = [i+1 for i in range(n_games)]
	plotLearning(x, ddqn_scores, eps_history, filename)