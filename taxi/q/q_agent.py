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
            "n_iter": 1000}        # Number of iterations
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
        success = 0
        for t_episode in range(config["n_iter"]):
                obs = env.reset()
                while True:
                        action = self.act(obs, config["eps"])
                        obs2, reward, done, prob = env.step(action)
                        future = np.max(q[obs2])
                        if done and reward == 0.0:
                                reward = -1.0
                        value = self.config["learning_rate"] * (q[obs][action] - reward - config["discount"] * future)
                        q[obs][action] -= value
                        if not done:
                                obs = obs2
                        else:
                                if reward == 1.0:
                                        success += 1
                                break
                if t_episode % 100 == 0:
                        print("sucess rate is {}".format(success))
                        success = 0

        for s in q.keys():
                print("state: {} {}".format(s, q[s]))
