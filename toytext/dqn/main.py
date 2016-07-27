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

t = Trainer()


t.train(q, env, render=False)

