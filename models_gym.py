import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque, namedtuple


class RecurrentPolicyClassicControl(nn.Module):
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

class RecurrentPolicyDiscrete(nn.Module):
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
  """Atari policy uses convolution layers for observations"""
  def __init__(self, action_space, debug_act=False):
    # input space is (3 x 220 x 160)
    super().__init__()
    self.debug_act = debug_act
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(32)
    self.linear = nn.Linear(4480, action_space)
    d = 0.01
    for p in self.parameters():
      p.data.uniform_(-d, d)
  def _log_activation(self, act, name):
    if self.debug_act:
      print('%8s mean activation: %12.3f' % (name, act.abs().mean().data[0]))
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    self._log_activation(x, "conv1")
    x = F.relu(self.bn2(self.conv2(x)))
    self._log_activation(x, "conv2")
    x = F.relu(self.bn3(self.conv3(x)))
    self._log_activation(x, "conv3")
    x = F.relu(self.bn4(self.conv4(x)))
    self._log_activation(x, "conv4")
    x = x.view(1, -1)
    x = self.linear(x)
    self._log_activation(x, "linear")
    return x