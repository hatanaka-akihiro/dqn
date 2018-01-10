from poker2 import Poker

import argparse
from time import sleep

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

parser = argparse.ArgumentParser(description='...')
parser.add_argument('--input', default=None)
parser.add_argument('--output', default='model_param.hdf5')
args = parser.parse_args()

env = Poker()
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1, 10)))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())

if args.input != None:
	model.load_weights(args.input)
	print("load: %s" % args.input)
	sleep(3)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=20000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.9)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=100, nb_steps_warmup=100, train_interval=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, nb_max_episode_steps=300)

model.save_weights(args.output)

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