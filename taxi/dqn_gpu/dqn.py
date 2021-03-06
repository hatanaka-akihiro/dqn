import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer import cuda
import os
from collections import namedtuple


class Q(chainer.Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    ENV_NAME = 'Taxi-v1'
    D = 500 
    hidden = 64

    def __init__(self, hidden, action_count):
        cuda.init()
        self.hidden = hidden
        self.action_count = action_count
        super(Q, self).__init__(
            l1=L.Linear(self.D, hidden, wscale=np.sqrt(hidden)).to_gpu(),
            l2=L.Linear(hidden, hidden, wscale=np.sqrt(hidden)).to_gpu(),
            l3=L.Linear(hidden, action_count, wscale=np.sqrt(action_count)).to_gpu()
        )

    def clone(self):
        return Q(self.hidden, self.action_count)

    def clear(self):
        self.loss = None

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        qv = self.l3(h2)
        return qv


class Agent():

    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions

    def action(self, qv, force_greedy=False):
        """ This is Agent's policy """
        is_greedy = True
        if np.random.rand() < self.epsilon and not force_greedy:
            action = self.actions[np.random.randint(0, len(self.actions))]
            is_greedy = False
        else:
            action = np.argmax(qv).__int__()
            #print(action)
        #print("action type:{} is_greedy:{}".format(type(action), is_greedy))
        return action, is_greedy


class Trainer():
    def __init__(self,
                    gamma=0.99,
                    memory_size=1000,
                    batch_size=10000,
                    learning_rate=1e-4,
                    decay_rate=0.99,
                    initial_epsilon=0.2,
                    epsilon_decay=1.0/10**3,
                    minimum_epsilon=0.1):
        self.gamma = gamma  # discount factor for reward
        self.memory_size = memory_size
        self.batch_size = batch_size  # memory size for experience replay
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=decay_rate)

    @classmethod
    def model_path(cls):
        return os.path.join(os.path.dirname(__file__), "model.backup")

    @classmethod
    def act(cls, observation, q_model, agent, prev=None):
        #s, merged = cls._make_input(observation, prev)
        s = cls._make_input(observation)
        qv = q_model.forward(chainer.Variable(cuda.to_gpu(np.array([s]))))
        action, is_greedy = agent.action(qv.data.flatten())
        return s, action, is_greedy, np.max(qv.data)

    @classmethod
    def print_model(cls, observation, q_model):
        s = cls._make_input(observation)
        qv = q_model.forward(chainer.Variable(cuda.to_gpu(np.array([s]))))
        print("{}={}".format(observation, qv.data))

    @classmethod
    def _make_input(cls, observation):
        s = cls._adjust(observation) if observation is not None else cls._make_array()
        return s

    @classmethod
    def _adjust(cls, s):
        I = cls._make_array()
        I[s] = 1.0
        return I

    @classmethod
    def _make_array(cls):
        return np.zeros(Q.D, dtype=np.float32)

    def calculate_loss(self, q_model, teacher, states, actions, next_states, rewards, dones):
        indices = np.random.permutation(len(states))  # random sampling
        shuffle = lambda x : np.array(x)[indices]
        states = shuffle(states)
        actions = shuffle(actions)
        next_states = shuffle(next_states)
        rewards = shuffle(rewards)
        dones = shuffle(dones)

        v_states = chainer.Variable(cuda.to_gpu(states))
        v_next_states = chainer.Variable(cuda.to_gpu(next_states))

        qv = q_model.forward(v_states)
        max_qv_next = np.max(teacher.forward(v_next_states).data, axis=1)
        target = qv.data.copy()
        teacher_qv = np.sign(rewards)
        for i, action in enumerate(actions):
            if dones[i]:
                teacher_qv[i] += self.gamma * max_qv_next[i]
            target[i, action] = teacher_qv[i]

        td = chainer.Variable(target) - qv
        #td_clip = td.data * (abs(td.data) <= 1) + 1.0 * (abs(td.data) > 1)
        #td_clip = chainer.Variable(td_clip.astype(np.float32))
        zeros = chainer.Variable(cuda.to_gpu(np.zeros(td.data.shape, dtype=np.float32)))
        #loss = F.mean_squared_error(td_clip, zeros)
        loss = F.mean_squared_error(td, zeros)
        return loss


    def train(self, q_model, env, render=False):
        """
        q model is optimized in accordance with openai gym environment. each action is decided by given agent
        """

        # setting up environment
        observation = env.reset()
        prev = None
        self.optimizer.setup(q_model)
        agent = Agent(self.initial_epsilon, list(range(env.action_space.n)))
        teacher = q_model.clone()

        # memory
        step = 0
        ss, acs, ns, rs, dones = [], [], [], [], []
        episode_count = 0
        episode_reward = 0
        total_reward = 0
        running_reward = None

        while True:
            if render: env.render()
            s, a, is_g, q_max = self.act(observation, q_model, agent, prev)
            prev = observation
            step += 1

            # execute action and get new observation
            observation, reward, done, info = env.step(a)
            #print("action={0} by {1}. reward={2}".format(a, "greedy(qvalue={0})".format(q_max) if is_g else "random", reward))
            episode_reward += reward

            # momory it
            next = self._make_input(observation)
            if len(ss) <= self.memory_size:
              ss.append(s)
              acs.append(a)
              ns.append(next)
              rs.append(reward)
              dones.append(done)
            else:
              index = step % self.memory_size
              ss[index] = s
              acs[index] = a
              ns[index] = next
              rs[index] = reward
              dones[index] = done

            if step % 100 == 0:
              self.optimizer.zero_grads()
              loss = self.calculate_loss(q_model, teacher, ss, acs, ns, rs, dones)
              loss.backward()
              self.optimizer.update()

            if step % self.batch_size == 0:
                teacher.copyparams(q_model)
                step = 0
                print("q_model update")
                serializers.save_npz(self.model_path(), q_model)

            if done:
                episode_count += 1
                total_reward += episode_reward
                print("episode {} has done. reward is {}. average reward is {}. step is {}."
                    .format(episode_count, episode_reward, total_reward / episode_count, step))
                episode_reward = 0

                # update policy
                agent.epsilon -= self.epsilon_decay
                if agent.epsilon < self.minimum_epsilon:
                    agent.epsilon = self.minimum_epsilon

                observation = env.reset() # reset env
                prev = None

