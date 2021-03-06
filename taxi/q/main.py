import gym
from q_agent import QAgent

env = gym.make('Taxi-v1')
print(env.action_space)
print(env.observation_space)

agent = QAgent(env.observation_space, env.action_space);
agent.learn(env)
total_reward = 0
total_step = 0
for i_episode in range(100):
	episode_reward = 0
	episode_step = 0
        observation = env.reset()
        while True:
                #env.render()
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
		episode_reward += reward
		episode_step += 1
                if done:
                        print("Episode {} has done. reward is {}. step is {}.".format(i_episode, episode_reward, episode_step))
			total_reward += episode_reward
			total_step += episode_step
			break
print("Average Reward is {}. Step is {}."
	.format(total_reward / 100, total_step / 100))
