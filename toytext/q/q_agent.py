import numpy as np
from collections import defaultdict

class QAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 100000}        # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(lambda: [0.0]*self.action_n)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
	if np.random.random() > eps:
        	action = np.argmax(self.q[observation])
	else:
		action = self.action_space.sample()
        return action

    def learn(self, env):
        config = self.config
        q = self.q
	total_reward = 0
	total_step = 0
	for t_episode in range(config["n_iter"]):
		obs = env.reset()
		episode_reward = 0
		episode_step = 0
        	while True:
            		action = self.act(obs, config["eps"])
            		obs2, reward, done, prob = env.step(action)
                	future = np.max(q[obs2])
			episode_reward += reward
			episode_step += 1
			#env.render()
			#print("reward was {}. episode_reward was {}".format(reward, episode_reward))
			if done:
				value = self.config["learning_rate"] * (q[obs][action] - episode_reward)
			else:
				value = self.config["learning_rate"] * (q[obs][action] - episode_reward - config["discount"] * future)

            		q[obs][action] -= value
            		if not done:
            			obs = obs2
			else:
				#print("episode {} has done. episode step is {}. reward is {}.".format(t_episode, episode_step, episode_reward))
				total_reward += episode_reward
				total_step += episode_step
				break
		if t_episode % 100 == 99:
			print("episode {} has done. average reward is {}. average step is {}.".format(t_episode, total_reward / 100, total_step / 100))
			total_reward = 0
			total_step = 0

	#for s in q.keys():
	#	print("state: {} {}".format(s, q[s]))
