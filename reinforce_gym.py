# import train
# from learners.base import BaseLearner
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque, namedtuple
import string
import os

import gym
from gym.wrappers import Monitor

cuda = torch.cuda.is_available()
cuda = False
print('cuda:', cuda)

long_t = torch.LongTensor if not cuda else torch.cuda.LongTensor
float_t = torch.FloatTensor if not cuda else torch.cuda.FloatTensor

alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
alphabet_l2i = {a: i for i, a in enumerate(alphabet)}
alphabet_i2l = {i: a for i, a in enumerate(alphabet)}

vocab = len(alphabet)

class SRPolicy(nn.Module):
  class State(namedtuple('State', ['obs'])):
    def detach(self):
      return type(self)(Variable(self.obs.data))

  def __init__(self, observation_space: int, action_space: int):
    super().__init__()
    self.obs_hidden_size = 64
    self.obs_input_size = observation_space
    self.obs_embedding_size = observation_space
    self.obs_n_layers = 4
    self.input_size = self.obs_embedding_size
    self.action_space = action_space
    self.obs_rnn = nn.GRU(self.input_size, self.obs_hidden_size, num_layers=self.obs_n_layers, dropout=0.5)
    self.affine = nn.Linear(self.obs_hidden_size, self.action_space)
    # d = 0.1
    # for p in self.parameters():
    #   p.data.uniform_(-d, d)
  def forward(self, input, state):
    # obs = self.embedding(input).unsqueeze(0)
    obs = input.unsqueeze(0)
    obs, new_obs_state = self.obs_rnn(obs, state.obs)
    obs = obs.squeeze(0)
    state = self.State(new_obs_state)
    linear = self.affine(obs)
    return linear, state

  def init_state(self):
    obs_hs = float_t(self.obs_n_layers, 1, self.obs_hidden_size).zero_()
    if cuda:
      obs_hs = obs_hs.cuda()
    return self.State(Variable(obs_hs))

class RecurrentPolicy(nn.Module):
  class State(namedtuple('State', ['obs', 'aux'])):
    def detach(self):
      return type(self)(Variable(self.obs.data), Variable(self.aux.data))

  def __init__(self, observation_space: int, action_space: int):
    super().__init__()
    self.obs_hidden_size = 128
    self.obs_input_size = observation_space
    self.obs_embedding_size = observation_space
    self.aux_input_size = 1
    self.aux_hidden_size = 16
    self.obs_n_layers = 1
    self.aux_n_layers = 1
    self.input_size = self.aux_hidden_size + self.obs_embedding_size
    self.action_space = action_space

    # self.embedding = nn.Embedding(self.obs_input_size, self.obs_embedding_size)
    self.obs_rnn = nn.GRU(self.input_size, self.obs_hidden_size, num_layers=self.obs_n_layers, dropout=0.5)
    self.aux_rnn = nn.GRU(self.aux_input_size, self.aux_hidden_size, num_layers=self.aux_n_layers, dropout=0.5)

    # aux RNN observes previous reward history
    self.affine = nn.Linear(self.obs_hidden_size, self.action_space)

    d = 0.1
    for p in self.parameters():
      p.data.uniform_(-d, d)

  def forward(self, input, prev_reward, state):
    # obs = self.embedding(input).unsqueeze(0)
    obs = input.unsqueeze(0)
    r = prev_reward.unsqueeze(0).unsqueeze(0)
    aux, new_aux_state = self.aux_rnn(r, state.aux)
    obs = torch.cat((obs, aux), 2)
    obs, new_obs_state = self.obs_rnn(obs, state.obs)
    obs = obs.squeeze(0)
    state = self.State(new_obs_state, new_aux_state)
    linear = self.affine(obs)
    return linear, state

  def init_state(self):
    obs_hs = float_t(self.obs_n_layers, 1, self.obs_hidden_size).zero_()
    aux_hs = float_t(self.aux_n_layers, 1, self.aux_hidden_size).zero_()
    if cuda:
      obs_hs = obs_hs.cuda()
      aux_hs = aux_hs.cuda()
    return self.State(Variable(obs_hs), Variable(aux_hs))

class LinearPolicy(nn.Module):
  def __init__(self, observation_space: int, action_space: int):
    super().__init__()
    self.affine = nn.Linear(observation_space, action_space)
  def forward(self, input):
    obs = input.unsqueeze(0)
    return self.affine(input)

class AtariPolicy(nn.Module):
  def __init__(self, action_space):
    # input space is (3 x 220 x 160)
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.linear = nn.Linear(4480, action_space)
    d = 0.1
    for p in self.parameters():
      p.data.uniform_(-d, d)
  def forward(self, obs):
    x = obs
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = x.view(1, -1)
    x = self.linear(x)
    x = F.sigmoid(x)
    return x

def describe_module_parameters(module):
  print(' ->> parameter norms')
  for name, node in module.state_dict().items():
    print('%20s %7.3f' % (name, node.norm()))
  print('\n')

def clip_grad(policy, max_grad_norm=100.0):
  grad_norm = 0.0
  for p in policy.parameters():
    grad_norm += p.grad.data.norm() ** 2
  grad_norm = grad_norm ** 0.5
  if grad_norm > max_grad_norm:
    grad_clip_ratio = max_grad_norm / grad_norm
    for p in policy.parameters():
      p.grad.mul_(grad_clip_ratio)
  return grad_norm

def _default_observation_preprocess_func(obs):
  """ndarray obs"""
  return float_t(np.expand_dims(obs, 0))

def _default_reward_preprocess_func(reward):
  """scalar reward"""
  return float_t([reward])

def train(env, policy, opt,
          observation_preproc_func=_default_observation_preprocess_func,
          reward_preproc_func=_default_reward_preprocess_func,
          episode_callback=None,
          feed_reward=False,
          max_reward=5000, max_episodes=10000,
          checkpoint_filename='checkpoint.tar', render=False,
          reward_baseline_len=30,
          save_frequency=100, max_grad_norm=10.0):
  episode = 0
  reward_history = []

  if os.path.isfile(checkpoint_filename):
    checkpoint = torch.load(checkpoint_filename)
    policy.load_state_dict(checkpoint['state_dict'])
    reward_history = checkpoint['reward_history']
    episode = checkpoint['episode']

  def save():
    torch.save({'state_dict': policy.state_dict(),
                'reward_history': reward_history,
                'episode': episode},
               checkpoint_filename)
    print('saved')

  stop = False

  has_state = hasattr(policy, 'init_state')

  for episode in range(episode, max_episodes):
    if has_state:
      policy_state = policy.init_state()
    state = env.reset()
    if render:
      env.render()
    total_reward = 0
    reward = 0
    done = False
    actions = []
    while not done:
      x = Variable(observation_preproc_func(state))
      r = Variable(reward_preproc_func(reward))
      args = [x]
      if feed_reward:
        args = args + [r]
      if has_state:
        args = args + [policy_state]
        act_linear, policy_state = policy(*args)
      else:
        act_linear = policy(*args)
      # print('act_linear', act_linear)
      sm = F.softmax(act_linear)
      # print('softmax', sm)
      act = sm.multinomial()
      actions.append(act)
      action = act.data[0][0]
      state, reward, done, what = env.step(action)
      # print(state, reward, done, what)
      total_reward += reward
      if render:
        env.render()
      if total_reward > max_reward:
        stop = True
        done = True
        print('solved!')
        break

    reward_history.append(total_reward)

    scaled_reward = None
    grad_norm = None

    if len(reward_history) > 1:
      rs = np.array(reward_history[-reward_baseline_len:])
      r_mean = rs.mean()
      r_std = rs.std()
      if r_std != 0:
        scaled_reward = (total_reward - r_mean) / r_std
      else:
        scaled_reward = 0

      opt.zero_grad()
      for act in actions:
        act.reinforce(scaled_reward)
      # import IPython; IPython.embed()
      autograd.backward(actions, [None for _ in actions])
      grad_norm = clip_grad(policy, max_grad_norm)
      # if grad_norm < 0.001:
      #   print('rel-r: %s, grad norm: %s' % (scaled_reward, grad_norm))
      #   import IPython; IPython.embed()
      opt.step()

      if episode % save_frequency == 0:
        save()

    print('episode %s, reward: %s, rel-r: %s, grad norm: %s (max %s)' % (episode, total_reward, scaled_reward, grad_norm, max_grad_norm))

    if episode_callback:
      episode_callback(policy, total_reward)

    if stop:
      save()
      return reward_history


def atari_observation_preprocess(frame):
  x = np.expand_dims(frame.astype(np.float).transpose(2, 0, 1), 0)
  # import ipdb; ipdb.set_trace()
  return float_t(x)

if __name__ == '__main__':
  # env = gym.make('Breakout-v0')
  env = gym.make('CartPole-v1')

  try:

    obs_shape = env.observation_space.shape
    if len(obs_shape) == 1:
      obs_space = obs_shape[0]

    action_space = env.action_space.n

    reward_history = []

    policy = LinearPolicy(obs_space, action_space)
    observation_preproc_func = _default_observation_preprocess_func

    # policy = AtariPolicy(action_space)
    # observation_preproc_func = atari_observation_preprocess


    if cuda:
      policy = policy.cuda()

    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    def episode_callback(policy, reward):
      describe_module_parameters(policy)

    reward_history = train(env, policy, opt,
      checkpoint_filename='checkpoint.tar',
      max_grad_norm=np.inf,
      observation_preproc_func=observation_preproc_func,
      episode_callback=episode_callback,
      reward_baseline_len=30, render=False)

    # canterpole
    # max grad norm 10, history reward baseline 100 -- no convergence after 1000 episodes
    # max grad norm 100, history reward baseline 30 -- solved after 603 episodes
    # max grad norm 100, history reward baseline 10 -- no convergence after 1500 episodes
    # max grad norm 10, history reward baseline 30, linear 4x20x2 policy -- solved after 1539

  finally:
    env.close()