import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import itertools
import argparse
from torch.distributions import Categorical
import math


class Actor(nn.Module):
    def __init__(self, params):
        super(Actor, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.o_dim + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 30)

    def forward(self, input):
        x = self.convert_type(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logist = self.out(x)
        return logist

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.params.device)
        return x


class Critic(nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(params.s_dim + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, input):
        x = self.convert_type(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.params.device)
        return x


class Agent():
    def __init__(self, params):
        self.params = params
        self.Actor = Actor(params).to(params.device)
        self.Critic = Critic(params).to(params.device)
        self.optim = torch.optim.Adam(itertools.chain(self.Actor.parameters(), self.Critic.parameters()), lr=params.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5000, gamma=0.95, last_epoch=-1)

    def choose_action(self, o, w):
        o = torch.Tensor(o)
        w = torch.Tensor(w)
        input = torch.cat([o, w])
        logist = self.Actor(input)
        prob1, prob2, prob3 = F.softmax(logist[0:10]), F.softmax(logist[10:20]), F.softmax(logist[20:30])
        self.dist1 = Categorical(prob1)
        self.dist2 = Categorical(prob2)
        self.dist3 = Categorical(prob3)
        self.act1 = self.dist1.sample()
        self.act2 = self.dist2.sample()
        self.act3 = self.dist3.sample()
        act = [self.act1.numpy(), self.act2.numpy(), self.act3.numpy()]
        return act

    def learn(self, r, w, ave_r):
        scalarized_reward = r[0] * w[0] + r[1] * w[1] - ave_r\
                            - 0.1 * math.sqrt(r[0] ** 2 + r[1] ** 2)
        log_prob1 = self.dist1.log_prob(self.act1)
        log_prob2 = self.dist2.log_prob(self.act2)
        log_prob3 = self.dist3.log_prob(self.act3)
        loss = -log_prob1 * scalarized_reward
        loss += -log_prob2 * scalarized_reward
        loss += -log_prob3 * scalarized_reward
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_dim', default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lr', default=0.001)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    agent = Agent(args)
    agent.choose_action(torch.Tensor([1,2,3]), torch.Tensor([0.5, 0.5]))
    agent.learn([1,1], [0.1,0.9])
