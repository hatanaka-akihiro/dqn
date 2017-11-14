import numpy as np
import gym
import gym.spaces

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class Deck():
	CARDS = []

	def __init__(self):
		self.reset()
	
	def getCards(num):
		cards = [];
		for i in range(num):
			cards.push(getCard())
		return arrayToNum(cards)

	def getCard():
		while(true):
			card = np.rand(len(CARDS))
			if CARDS[card] == 0:
				CARDS[card] == 1
				return card
	
	def reset():
		CARDS = np.zeros(13 * 4)

	def draw(cards, changes):
		cards = numToArray(cards)
		changes = numToArray(changes)	
		for i in changes:
			cards[i] = getCard()
		return arrayToNum(cards)

	def getScore(cards):
		cards = numToArray(cards)
		pears = np.zeros(13)
		for i in cards:
			num = i % 13
			pears[num]++
		score = 0
		for i in pears:
			if i > 1:
				score+=i*10
		return score

        def numToArray(num, length):
                cards = [];
                for i in range(length).reverse():
                        if num >= 2^i:
                                cards.push(i)
                                num -= 2^i
1               return cards;

	def arrayToNum(array, length):
		num = 0
		for i in array:
			num += 2^i
		return num


# 直線上を動く点の速度を操作し、目標(原点)に移動させることを目標とする環境
class Poker(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2^5) 
        self.observation_space = gym.spaces.Discrete(2^62) #62C5
        self._deck = Deck()

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        self._cards = self._deck.draw(self._cards, action) 

	reward = self._deck.getScore(self._cards)
	done = true
        return self._cards, reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):
        # 初期stateは、位置はランダム、速度ゼロ
	self._deck.reset()
        self._cards = self._deck.getCards(5)
	return self._cards

env = Poker()
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=300)

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=10, visualize=False, callbacks=[cb_ep])


import matplotlib.pyplot as plt

for obs in cb_ep.observations.values():
    plt.plot([o[0] for o in obs])
plt.xlabel("step")
plt.ylabel("pos")
