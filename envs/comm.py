import argparse
import numpy as np
import random
import torch
from gym.spaces import Box, Discrete
# NUM_UE = 10
NUM_Channel = 1
LAMBDA = 3
NEW_TASK_MEAN = 500000000
NEW_TASK_VAR = 100000
# B = 1000000000/NUM_UE# 每个用户的信道带宽
var_noise = 10**(-5)
DELTA_T = 0.01
T_UNIT = 0.01
np.random.seed(5)

class CommEnv():
    def __init__(self, args=None ,eval=False):
        args = args
        self.args = args
        self.num_user = args.num_user
        self.fe = args.fe
        self.fc = args.fc
        self.alpha = args.alpha
        self.beta = args.beta
        self.T_max = args.T_max
        self.discrete = args.discrete
        if self.discrete:
            self.ACTIONS = [i * 0.1 for i in range(11)]
        self.eval = eval
        self.B = 1000000000/self.num_user

    def init_channel_matrix(self):
        # UE_Channel_matrix = np.zeros((NUM_Channel, self.num_user))
        # for j in range(self.num_user):
        #     if j < NUM_Channel:
        #         UE_Channel_matrix[j, j] = 1
        #     else:
        #         UE_Channel_matrix[int(j - NUM_Channel), j] = 1
        # UE_Channel_matrix = np.array(UE_Channel_matrix)
        # return UE_Channel_matrix
        '''
        modified!!!!!
        :return: 
        '''
        return np.ones((NUM_Channel, self.num_user))

    def reset(self):
        self.n_step = 0
        self.ep_r = np.array([0, 0])
        self.task_remain = np.zeros(self.num_user)
        self.UE_Channel_matrix = self.init_channel_matrix()
        self.create_new_task(DELTA_T)
        np.random.seed(5)
        self.He = abs(1 / np.sqrt(2) * (np.random.randn(self.num_user)
                                        + 1j * np.random.randn(self.num_user)))  # 边缘
        self.Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(self.num_user)
                                              + 1j * np.random.randn(self.num_user)))  # 云
        print(self.He, self.Hc)
        task_remain = (self.task_remain - self.args.mean_normal) / self.args.var_normal / 1000
        obs = np.array([list(task_remain[i].reshape(-1)) + [self.He[i]]
                        + [self.Hc[i]] for i in range(self.num_user)])

        return obs

    def sum_rate(self, UE_Channel_matrix, He, Hc, pe, pc, B, var_noise):
        pe = np.array(pe)
        pc = np.array(pc)
        He = np.array(He)
        Hc = np.array(Hc)
        rate_edge = np.zeros((self.num_user))
        rate_cloud = np.zeros((self.num_user))
        # 边缘网络

        Ie = 0; Ic = 0
        for n in range(self.num_user):
            Ie += He[n] ** 2 * pe[n]
            Ic += Hc[n] ** 2 * pc[n]

        for n in range(self.num_user):
            rate_edge[n] = B * np.math.log2(1 + He[n] ** 2 * pe[n] / (var_noise + Ie - He[n] ** 2 * pe[n]))
            rate_cloud[n] = B * np.math.log2(1 + Hc[n] ** 2 * pc[n] / (var_noise + Ic - Hc[n] ** 2 * pc[n]))

        # for n in range(NUM_Channel):
        #     U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
        #     L = len(U)
        #     for m in range(L):
        #         if L > 1:
        #             Com_H = He[n, U[m]]
        #             # 串行干扰消除干扰用户计算
        #             I1 = np.zeros(L)
        #             I2 = np.zeros(L)
        #             for m1 in range(L):  # 依次检验所有用户
        #                 if Com_H <= He[n, U[m1]]:  # 大信号 包含原信号
        #                     I1[m1] = He[n, U[m1]] ** 2 * pe[U[m1]]
        #                 else:
        #                     I2[m1] = 0.01 * He[n, U[m1]] ** 2 * pe[U[m1]]
        #             rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - He[n, U[m]] ** 2 * pe[U[m]]))
        #         elif L == 1:
        #             rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / var_noise)

        # 云网络
        # for n in range(NUM_Channel):
        #     U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
        #     L = len(U)
        #     for m in range(L):
        #         if L > 1:
        #             Com_H = Hc[n, U[m]]
        #             I1 = np.zeros(L)
        #             I2 = np.zeros(L)
        #             for m1 in range(L):
        #                 if Com_H <= Hc[n, U[m1]]:
        #                     I1[m1] = Hc[n, U[m1]] ** 2 * pc[U[m1]]
        #                 else:
        #                     I2[m1] = 0.01 * Hc[n, U[m1]] ** 2 * pc[U[m1]]
        #             rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - Hc[n, U[m]] ** 2 * pc[U[m]]))
        #         elif L == 1:
        #             rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / var_noise)

        return rate_edge, rate_cloud

    def compute_reward(self, UE_Channel_matrix, task_coef, pe, pc, task_current):
        task_coef = [1 - task_coef[i] for i in range(self.num_user)]
        E_off = np.zeros([self.num_user])
        E_exe = np.zeros([self.num_user])
        E = np.zeros([self.num_user])
        T = np.zeros([self.num_user])

        rate_edge, rate_cloud = self.sum_rate(UE_Channel_matrix, self.He, self.Hc, pe, pc, self.B, var_noise)
        rate_edge = rate_edge + 0.1
        rate_cloud = rate_cloud + 0.1
        for j in range(self.num_user):
            # i = np.transpose(np.nonzero(UE_Channel_matrix[:,j]))[0][0]
            E_off[j] = (pe[j] * task_coef[j] * task_current[j]) / rate_edge[j] + (pc[j] * (1 - task_coef[j]) * task_current[j]) / rate_cloud[j]
            Te = (task_coef[j] * task_current[j]) / rate_edge[j] + (self.alpha * task_coef[j] * task_current[j]) / self.fe
            Tc = ((1 - task_coef[j]) * task_current[j]) / rate_cloud[j] + (self.alpha * (1 - task_coef[j]) * task_current[j]) / self.fc
            E_exe[j] = self.beta * (self.alpha * task_coef[j] * task_current[j] * self.fe ** 2 + self.alpha * (1- task_coef[j]) * task_current[j]* self.fc ** 2)
            E[j] = E_off[j] + E_exe[j]
            # print("Te:{0}\nTc:{1}\n".format(Te, Tc))
            T[j] = max(Te, Tc)
        return E, T

    def create_new_task(self, delta_t):
        for u in range(self.num_user):
            task_num = np.random.poisson(lam=self.args.lam, size=1)[0]
            for i in range(task_num):
                new_task = random.normalvariate(self.args.mean_normal, self.args.var_normal)
                self.task_remain[u] += new_task

        # return new_task

    def step(self, actions, delta_t=None):
        self.n_step += 1
        if delta_t == None:
            delta_t = DELTA_T
        delta_t += self.args.processing_period

        if self.discrete:
            x = [self.ACTIONS[i[0]] for i in actions]
            Pe = [self.ACTIONS[i[1]] for i in actions]
            Pc = [self.ACTIONS[i[2]] for i in actions]
        else:
            x = [actions[0]]
            Pe = [actions[1]]
            Pc = [actions[2]]

        for i in range(len(Pe)):
            if Pc[i] == 0:
                Pc[i] += 0.01
            if Pe[i] == 0:
                Pe[i] += 0.01

        E, T = self.compute_reward(self.UE_Channel_matrix, x, Pe, Pc, self.task_remain)
        reward = [-T.max(), -E.sum() * (2680/28)]
        self.task_remain = np.zeros_like(self.task_remain)
        self.create_new_task(delta_t)
        self.n_step += 1
        # self.He = abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user)
        #                                 + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 边缘
        # self.Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, self.num_user)
        #                                       + 1j * np.random.randn(NUM_Channel, self.num_user)))  # 云  #TODO
        task_remain = (self.task_remain - self.args.mean_normal) / self.args.var_normal / 1000
        obs = np.array([list(task_remain[i].reshape(-1)) + [self.He[i]]
                        + [self.Hc[i]] for i in range(self.num_user)])
        #TODO to be modified

        return obs, reward, False, {}

    def get_state(self):
        task_remain = (self.task_remain - self.args.mean_normal) / self.args.var_normal / 1000
        return np.array(list(task_remain.reshape(-1))
                        + list(self.He.reshape(-1)) + list(self.Hc.reshape(-1)))

    @property
    def observation_space(self):
        return Box(low=-float("inf"), high=float("inf"), shape=(2 + 1,))

    @property
    def action_space(self):
        return Discrete(11)

    @property
    def state_space(self):
        return Box(low=-float("inf"), high=float("inf"), shape=(2 * NUM_Channel * self.num_user + self.num_user,))



def get_args():
    parser = argparse.ArgumentParser(description="computation offloading environment")
    parser.add_argument('--fe', default=10**14)
    parser.add_argument('--fc', default=10**15)
    parser.add_argument('--alpha', default=10**8)
    parser.add_argument('--beta', default=10**(-46))
    parser.add_argument('--T_max', default=8)
    parser.add_argument('--lam', default=10)
    parser.add_argument('--mean_normal', default=100000)
    parser.add_argument('--var_normal', default=10000)
    parser.add_argument('--num_user', default=3)
    parser.add_argument('--processing_period', default=0.01)
    parser.add_argument('--discrete', default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    env = CommEnv(args)
    env.reset()
    for ep in range(10):
        tot_rew1 = 0
        tot_rew2 = 0
        for i in range(200):
            action = [[np.random.randint(0, 10) for _ in range(3)] for _ in range(args.num_user)]
            # print(action)
            action = [[5, 10, 10] for _ in range(args.num_user)]
            # print(action)
            # print("action:", action)
            # state, reward, done, info = env.step(action, 0.001)
            # print(info)
            o, r, _, _ = env.step(action)
            print("o:", o)
            # print(r)
            tot_rew1 += r[0]
            tot_rew2 += r[1]
        print("total reward:", tot_rew1, tot_rew2)
