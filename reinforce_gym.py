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

from models_gym import *

cuda = torch.cuda.is_available()
print('cuda:', cuda)

long_t = torch.LongTensor if not cuda else torch.cuda.LongTensor
float_t = torch.FloatTensor if not cuda else torch.cuda.FloatTensor

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
      sm = F.softmax(act_linear)
      act = sm.multinomial()
      actions.append(act)
      action = act.data[0][0]
      state, reward, done, what = env.step(action)
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


atari = True

if __name__ == '__main__':
  if atari:
    env = gym.make('Breakout-v0')
  else:
    env = gym.make('CartPole-v1')

  obs_shape = env.observation_space.shape
  if len(obs_shape) == 1:
    obs_space = obs_shape[0]
  action_space = env.action_space.n

  reward_history = []

  try:

    if not atari:
      policy = LinearPolicy(obs_space, action_space)
      observation_preproc_func = _default_observation_preprocess_func
    else:
      policy = AtariPolicy(action_space)
      observation_preproc_func = atari_observation_preprocess

    if cuda:
      policy = policy.cuda()

    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)

    def episode_callback(policy, reward):
      describe_module_parameters(policy)

    reward_history = train(env, policy, opt,
      checkpoint_filename='checkpoint.tar',
      max_grad_norm=10.0,
      observation_preproc_func=observation_preproc_func,
      episode_callback=episode_callback,
      reward_baseline_len=30,
      render=False)

    # canterpole
    # max grad norm 10, history reward baseline 100 -- no convergence after 1000 episodes
    # max grad norm 100, history reward baseline 30 -- solved after 603 episodes
    # max grad norm 100, history reward baseline 10 -- no convergence after 1500 episodes
    # max grad norm 10, history reward baseline 30, linear 4x20x2 policy -- solved after 1539

  finally:
    env.close()