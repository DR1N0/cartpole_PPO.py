import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 设置超参数
LR_A = 0.0003
LR_C = 0.0005
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
gamma = 0.9
epsilon = 0.1
epoch = 2
T_max = 500


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1_A = nn.Linear(4, 100)
        self.fc1_C = nn.Linear(4, 100)
        self.fc2_A = nn.Linear(100, 2)  # actor输出2维左右
        self.fc2_C = nn.Linear(100, 1)  # critic输出1维value
        self.optimizer_A = torch.optim.Adam(self.parameters(), lr=LR_A)
        self.optimizer_C = torch.optim.Adam(self.parameters(), lr=LR_C)

    # actor
    def pi(self, x, softmax_dim=0):
        x = torch.relu_(self.fc1_A(x))
        x = torch.tanh_(self.fc2_A(x))
        prob = nn.functional.softmax(x, dim=softmax_dim)
        return prob

    # critic
    def value(self, x):
        x = torch.relu_(self.fc1_C(x))
        v = self.fc2_C(x)
        return v

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def learn(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(epoch):
            td_target = r + gamma * self.value(s_prime) * done_mask  # 游戏结束则长期期望归零
            delta = td_target - self.value(s)
            delta = delta.detach().numpy()

            # 计算当前策略相比baseline的优势
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # 更新幅度限制
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            # 优化critic
            loss_C = surr
            self.optimizer_C.zero_grad()
            loss_C.mean().backward()
            self.optimizer_C.step()
            # 优化actor
            loss_A = nn.functional.smooth_l1_loss(self.value(s), td_target.detach())
            self.optimizer_A.zero_grad()
            loss_A.mean().backward()
            self.optimizer_A.step()


def get_action(prob, stability):
    pp = random.uniform(0, 1)
    a = int(torch.argmax(prob).numpy())
    if pp < stability:
        return 1 - a
    return a


def train(max_episode=50000):
    # 创建倒立摆环境
    env = gym.make('CartPole-v1')
    model = PPO()
    # model.state_dict = torch.load('cartpole_model.pth')
    score = 0.0
    score_episode = 0.0
    print_interval = 20
    score_lst, score_episode_lst = [], []

    # 主循环
    score_cnt = 0
    for n_epi in range(max_episode):
        s = env.reset()
        stability = 1 - np.power(0.3, 1 + n_epi / 3000)
        done = False
        while not done:
            for t in range(T_max):
                prob = model.pi(torch.from_numpy(s).float())
                a = get_action(prob, stability)
                s_prime, r, done, info = env.step(a)
                model.data.append(
                    (s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime
                score_episode += r
                score += r
                if done:
                    break
            score_lst.append(score)
            score = 0

            model.learn()
        # 打印成绩
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, score_episode / print_interval))
            score_episode_lst.append(score_episode / print_interval)
            if score_episode / print_interval == 500:
                score_cnt += 1
            score_episode = 0.0
        if score_cnt == 30:
            break
    torch.save(model.state_dict(), 'cartpole_model.pth')
    env.close()

    index1 = list(range(len(score_lst)))
    index2 = list(range(len(score_episode_lst)))
    for i in range(len(index2)):
        index2[i] *= print_interval
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.plot(index1, score_lst, 'o')
    plt.plot(index2, score_episode_lst, linewidth=2)
    plt.title('Score - Episode Relation')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()


if __name__ == '__main__':
    train(max_episode=5000)

#####此处仅为了视频演示时间不过长，设置max_episode为5000，实际训练数字可以大一个数量级。
