import gym
from q_agent import QAgent

env = gym.make('FrozenLake-v0')
print(env.action_space)
print(env.observation_space)

agent = QAgent(env.observation_space, env.action_space, n_iter=100000);
agent.learn(env)
success = 0
for i_episode in range(100):
        observation = env.reset()
        while True:
                #env.render()
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
                if done:
                        #print("Episode finished after {} timesteps".format(t+1))
                        if reward == 1.0:
                                success += 1
                        break
print("success rate is {}".format(success))
