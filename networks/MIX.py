import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class HyperNet(nn.Module):
    def __init__(self, args):
        super(HyperNet, self).__init__()
        self.args = args
        self.device = args.device
        self.hyper_w1 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * args.n_agents * args.hyper_h1),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, args.hyper_h1 * self.args.n_obj),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, args.hyper_h1)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.s_dim + args.n_obj, 128),
            nn.ReLU(),
            nn.Linear(128, self.args.n_obj)
        )
        self.to(self.device)

    def forward(self, s, w):
        s = self.convert_type(s)
        w = self.convert_type(w)
        x = torch.cat([s, w], dim=-1)
        w1 = self.hyper_w1(x).view(-1, 3 * self.args.n_agents, self.args.hyper_h1)
        w2 = self.hyper_w2(x).view(-1, self.args.hyper_h1, self.args.n_obj)
        b1 = self.hyper_b1(x).unsqueeze(1)
        b2 = self.hyper_b2(x).unsqueeze(1)

        return w1, w2, b1, b2

    def get_Q_tot(self, s, w, Q):
        s = s.unsqueeze(1)
        s = s.repeat([1, self.args.batch_size_p, 1]).view(-1, s.shape[-1])
        w = w.repeat([self.args.batch_size, 1]).view(-1, self.args.n_obj)
        Q = Q.unsqueeze(1)
        w1, w2, b1, b2 = self.forward(s, w)
        w1 = torch.abs(w1)
        w2 = torch.abs(w2)
        # b1 = torch.abs(b1)
        # b2 = torch.abs(b2)
        h1 = F.relu(torch.bmm(Q, w1) + b1)
        out = F.relu(torch.bmm(h1, w2) + b2)

        return out.squeeze(1)
        return torch.bmm(w.unsqueeze(1), out.permute(0, 2, 1)).squeeze(-1)

    def convert_type(self, input):
        if not isinstance(input, torch.Tensor):
            input = torch.Tensor(input)
        if input.device != torch.device(self.args.device):
            input = input.to(self.device)

        return input

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default=5)
    parser.add_argument("--gpu", default=False)
    parser.add_argument("--buffer_size", default=50000)
    parser.add_argument("--h_dim", default=128)
    parser.add_argument("--n_obj", default=2)
    parser.add_argument("--hyper_h1", default=64)
    parser.add_argument("--n_threads", default=1)
    parser.add_argument("--s_dim", default=5)
    parser.add_argument("--a_dim", default=10)

    return parser.parse_args()

if __name__ == '__main__':
    Q = HyperNet(get_args())
    t = Q.parameters()
    print()



