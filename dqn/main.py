import os
import gym
import chainer
from chainer import serializers
from dqn import Q, Trainer

env = gym.make('FrozenLake-v0')
print(env.action_space)
print(env.observation_space)

q = Q(Q.hidden, env.action_space.n)
path = Trainer.model_path()
if os.path.isfile(path):
        serializers.load_npz(path, q)

t = Trainer(gamma=0.99,
        memory_size=100,
        batch_size=100,
        learning_rate=1e-4,
        decay_rate=0.99,
        initial_epsilon=0.5,
        epsilon_decay=1.0/10**4,
        minimum_epsilon=0.1)


t.train(q, env, render=False)

