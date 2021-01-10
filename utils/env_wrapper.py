import torch
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_func, args):
    parent_remote.close()
    env = env_func(args)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space, env.state_space))
        elif cmd == 'get_state':
            remote.send(env.get_state())

class EnvWrapper():
    def __init__(self, args, env_func):
        self.args = args
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.args.n_threads)])
        self.processes = [Process(target=worker, args=(work_remote, remote, env_func, args))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.processes:
            p.daemon = True
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space, self.state_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        tup = [self.remotes[i].recv() for i in range(self.args.n_threads)]

        obs = [t[0] for t in tup]
        rew = [r[1] for r in tup]
        done = [d[2] for d in tup]
        info = [i[3] for i in tup]

        return obs, rew, done, info

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        # print([self.remotes[i].recv() for i in range(self.args.n_threads)])
        tup = [self.remotes[i].recv() for i in range(self.args.n_threads)]

        return tup

    def get_state(self):
        for remote in self.remotes:
            remote.send(("get_state", None))
        tup = [self.remotes[i].recv() for i in range(self.args.n_threads)]

        return tup

class EnvWrapper_sigle():
    def __init__(self, args, env_func):
        self.args = args
        self.env = env_func(args)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space

    def reset(self):
        tup = [self.env.reset()]
        return tup

    def step(self, actions):
        tup = self.env.step(actions[0])
        obs = [tup[0]]
        rew = [tup[1]]
        done = [tup[2]]
        info = [tup[3]]
        return obs, rew, done, info

    def get_state(self):
        tup = [self.env.get_state()]
        return tup



if __name__ == '__main__':
    from envs.comm import CommEnv
    from train import get_env_args
    import numpy as np
    env = EnvWrapper(get_env_args(), CommEnv)
    s = env.reset()
    action = [[[np.random.randint(0, 10) for _ in range(3)] for _ in range(10)] for _ in range(3)]
    o, r, d, i = env.step(action)
    print(o, r, d, i)