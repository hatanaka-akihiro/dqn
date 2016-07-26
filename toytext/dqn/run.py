import os
import sys
import gym
import numpy as np
import chainer
from chainer import serializers
from dqn import Q, Agent, Trainer


def main(game_count=20):
    path = Trainer.model_path()

    env = gym.make("FrozenLake-v0")
    q = Q(Q.hidden, env.action_space.n)
    agent = Agent(0.0, list(range(env.action_space.n)))
    serializers.load_npz(path, q)

    for i in range(Q.D):
        Trainer.print_model(i, q)

    for i_episode in range(game_count):
        observation = env.reset()
        prev = None
        step = 0
        while True:
            #env.render()
            s, a, _, _ = Trainer.act(observation, q, agent, prev)
            prev = s
            observation, reward, done, info = env.step(a)
            if done:
                print("episode {} has done. reward is {}. its length is {}.".format(i_episode, reward, step))
                break
            else:
                step += 1


if __name__ == "__main__":
    main()
