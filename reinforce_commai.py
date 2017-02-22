import train
from learners.base import BaseLearner
import numpy as np
import torch as th
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from collections import deque, namedtuple
import string

cuda = th.cuda.is_available()
cuda = False
print('cuda:', cuda)

long_t = th.LongTensor if not cuda else th.cuda.LongTensor
float_t = th.FloatTensor if not cuda else th.cuda.FloatTensor

alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
alphabet_l2i = {a: i for i, a in enumerate(alphabet)}
alphabet_i2l = {i: a for i, a in enumerate(alphabet)}

vocab = len(alphabet)

class PolicyState(namedtuple('PolicyState', ['obs', 'aux'])):
  def detach(self):
    return type(self)(Variable(self.obs.data), Variable(self.aux.data))

class Policy(nn.Module):
  def __init__(self):
    super().__init__()
    self.obs_hidden_size = 128
    self.obs_embedding_size = vocab
    self.aux_input_size = 1
    self.aux_hidden_size = 16
    self.obs_n_layers = 4
    self.aux_n_layers = 4
    self.input_size = self.aux_hidden_size + self.obs_embedding_size

    self.embedding = nn.Embedding(vocab, vocab)
    self.obs_rnn = nn.GRU(self.input_size, self.obs_hidden_size, num_layers=self.obs_n_layers, dropout=0.5)
    self.aux_rnn = nn.GRU(self.aux_input_size, self.aux_hidden_size, num_layers=self.aux_n_layers, dropout=0.5)
    # aux RNN observes previous reward history
    self.affine = nn.Linear(self.obs_hidden_size, vocab)

    d = 0.1
    for p in self.parameters():
      p.data.uniform_(-d, d)

  def forward(self, input, prev_reward, state):
    obs = self.embedding(input).unsqueeze(0)
    r = prev_reward.unsqueeze(0).unsqueeze(0)
    aux, new_aux_state = self.aux_rnn(r, state.aux)
    obs = th.cat((obs, aux), 2)
    obs, new_obs_state = self.obs_rnn(obs, state.obs)
    obs = obs.squeeze(0)
    state = PolicyState(new_obs_state, new_aux_state)
    linear = self.affine(obs)
    return linear, state

  def init_state(self):
    obs_hs = float_t(self.obs_n_layers, 1, self.obs_hidden_size).zero_()
    aux_hs = float_t(self.aux_n_layers, 1, self.aux_hidden_size).zero_()
    if cuda:
      obs_hs = obs_hs.cuda()
      aux_hs = aux_hs.cuda()
    return PolicyState(Variable(obs_hs), Variable(aux_hs))

Action = namedtuple('Action', ['action', 'reward'])

class Agent(BaseLearner):
  REINFORCE_STEP = 100
  def __init__(self):
    self.policy = Policy()
    if cuda:
      self.policy = self.policy.cuda()
    self.opt = th.optim.Adam(self.policy.parameters(), lr=1e-3)
    self.state = self.policy.init_state()
    self._prev_action = None
    self.prev_reward = 0.
    self.history = []
    self.step = 0

  def try_learning_step(self):
    if self.history and (len(self.history) >= self.REINFORCE_STEP):
      gamma = 1.0
      rewards = []
      cum_reward = 0.
      R = 0.0
      for h in self.history[::-1]:
        cum_reward += h.reward
        R = R * gamma + h.reward
        rewards.append(R)
        rewards.append(h.reward-0.01)
      rewards = np.array(rewards[::-1])
      rewards = (rewards - rewards.mean()) / rewards.std()

      actions = [h.action for h in self.history]

      for action, reward in zip(actions, rewards):
        action.reinforce(reward)

      self.opt.zero_grad()
      autograd.backward(actions, [None for _ in actions])

      grad_norm = 0.0
      for p in self.policy.parameters():
        grad_norm += p.grad.data.norm() ** 2
      grad_norm = grad_norm ** 0.5

      max_grad_norm = 5.0
      if grad_norm > max_grad_norm:
        grad_clip_ratio = max_grad_norm / grad_norm
        for p in self.policy.parameters():
          p.grad.mul_(grad_clip_ratio)

      print('learning | step %s: cum rev %s over %s steps | grad norm %.5f / clip to %.1f' % (self.step, cum_reward, len(actions), grad_norm, max_grad_norm))
      self.opt.step()
      self.history = []
      self.last_learning_step = self.step

  def try_reward(self, reward):
    if reward is None:
      reward = 0
    if self._prev_action is not None:
      self.history.append(Action(self._prev_action, reward))
      self.prev_reward = reward

  def next(self, input_char):
    self.try_learning_step()
    input_byte = alphabet_l2i[input_char]
    assert isinstance(input_byte, int)
    assert 0 <= input_byte < vocab

    x = Variable(long_t([input_byte]))
    r = Variable(float_t([self.prev_reward]))
    act_linear, self.state = self.policy(x, r, self.state)
    self.state = self.state.detach()
    act_dist = F.softmax(act_linear)
    max_l = act_linear.max().data[0]
    max_p = act_dist.max().data[0]

    act = act_dist.multinomial()
    self._prev_action = act
    out_byte = act.data[0][0]

    out_char = alphabet_i2l[out_byte]
    if self.step % 100 == 0:
      print('input[%s][%s], output[%s][%s], max-l %.5f, max-p %.5f' % (input_byte, input_char, out_byte, out_char, max_l, max_p))
    self.step += 1

    return out_char


if __name__ == '__main__':
  train.train_agent(Agent(), 0)