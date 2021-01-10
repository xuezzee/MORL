import torch
import numpy as np
from torch import nn
import random


class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.args = args
        self.device = args.device
        self.Linear = nn.Linear(args.o_dim+args.n_obj, args.h_dim)
        self.GRU1 = nn.Linear(args.h_dim, args.h_dim)
        self.GRU2 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.GRU3 = nn.GRUCell(args.h_dim, args.h_dim, bias=True)
        self.out1 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)
        self.out2 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)
        self.out3 = nn.Linear(args.h_dim, args.a_dim * args.n_obj)
        self.to(self.device)

    def forward(self, input):
        input = self.convert_type(input)
        h = self.Linear(input)
        h1 = self.GRU1(h)
        h2 = self.GRU2(h, h1)
        h3 = self.GRU3(h, h2)
        out1 = self.out1(h).view(-1, self.args.a_dim, self.args.n_obj)
        out2 = self.out2(h2).view(-1, self.args.a_dim, self.args.n_obj)
        out3 = self.out3(h3).view(-1, self.args.a_dim, self.args.n_obj)

        return [out1, out2, out3]

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if x.device != torch.device(self.args.device):
            x = x.to(self.args.device)
        return x

    def choose_action(self, o, w, epsilon):
        o = self.convert_type(o)
        w = self.convert_type(w)
        input = torch.cat([o, w], dim=-1).to(self.device)
        q = self.forward(input)
        for i in range(len(q)):
            q[i] = torch.cat([torch.matmul(w[t].view(-1, self.args.n_obj), q[i][t].view(-1, self.args.n_obj).T)
                              for t in range(self.args.n_threads)])
        actions = []
        for i in range(3):
            if random.random() < epsilon:
                actions.append(np.array([random.randint(0, self.args.a_dim - 1)
                                         for _ in range(self.args.n_threads)]))
            else:
                actions.append(torch.argmax(q[i], dim=-1).data.cpu().numpy())

        return actions

    def get_target_q(self, x, w):
        x = self.convert_type(x)
        w = self.convert_type(w)
        x = x.reshape((-1, self.args.a_dim + self.args.n_obj))
        q = self.forward(x)
        for i in range(len(q)):
            q[i] = q[i].reshape((-1, self.args.batch_size_p*self.args.a_dim, self.args.n_obj))
        q = [torch.matmul(q[i], w.unsqueeze(-1)) for i in range(len(q))]

        return torch.cat([torch.max(q[i], dim=-2).values.unsqueeze(0) for i in range(len(q))])

    def get_q(self, x, w, a):
        x = self.convert_type(x)
        w = self.convert_type(w)
        a = self.convert_type(a)
        a = a.expand(self.args.batch_size_p, a.shape[0], a.shape[1]).permute(2, 1, 0).unsqueeze(-1)
        a = a.type(torch.int64).repeat([1, 1, 1, self.args.n_obj]).unsqueeze(-2)
        input = torch.cat([x, w], dim=-1)
        q = self.forward(input)
        q = [q[i].reshape(self.args.batch_size_p, self.args.batch_size, self.args.a_dim, self.args.n_obj).\
                                                permute(1, 0, 2, 3) for i in range(3)]
        q = [torch.gather(q[i], -2, a[i]) for i in range(3)]
        wq = [torch.bmm(w.view(-1, 1, self.args.n_obj), q[i].view(-1, self.args.n_obj, 1)).squeeze(-1) for i in range(3)]
        wq = torch.cat(wq, dim=-1)
        return wq


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
    Q = Q_net(get_args())
    t = Q.parameters()
    print()


