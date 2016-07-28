import os
import sys
import gym
import numpy as np
import chainer
from chainer import serializers
from dqn import Q, Agent, Trainer


def main(game_count=20):
    path = Trainer.model_path()

    env = gym.make(Q.ENV_NAME)
    q = Q(Q.hidden, env.action_space.n)
    agent = Agent(0.0, list(range(env.action_space.n)))
    serializers.load_npz(path, q)

    for i in range(Q.D):
        Trainer.print_model(i, q)

    for i_episode in range(game_count):
        observation = env.reset()
        prev = None
        step = 0
	episode_reward = 0
        while True:
            #env.render()
            s, a, _, _ = Trainer.act(observation, q, agent, prev)
            prev = s
            observation, reward, done, info = env.step(a)
	    episode_reward += reward
            step += 1
            if done:
                print("episode {} has done. reward is {}. step is {}."
			.format(i_episode, episode_reward, step))
                break
	    elif step >= 1000:
		print("episode {} has faield. reward is {}. step is {}."
			.format(i_episode, episode_reward, step))
		break

if __name__ == "__main__":
    main()
