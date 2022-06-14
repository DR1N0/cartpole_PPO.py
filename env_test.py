from model_train import PPO, get_action
import torch
import gym
from gym import wrappers
import matplotlib.pyplot as plt

def eval():
    model = PPO()
    model.load_state_dict(torch.load('cartpole_model_trained.pth'))
    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env, "./cartpole",video_callable=False, force=True)
    # 主循环
    count = 100
    stability = 0.95
    show_interval = 10
    score_total = 0
    score_interval = 0
    score_lst = []
    for n_epi in range(count):
        score = 0
        s = env.reset()
        done = False
        while not done:
            cnt = 0
            prob = model.pi(torch.from_numpy(s).float())
            a = get_action(prob, stability = stability)
            s_prime, r, done, info = env.step(a)
            #env.render()
            s = s_prime
            score += r
            if cnt > 1000:
                break
        if n_epi == 0:
            continue
        score_lst.append(score)
        score_interval += score
        score_total += score
        if n_epi % show_interval == 0:
            print('avg score of #', n_epi - show_interval, '-', n_epi, ':', score_interval/show_interval)
            score_interval = 0
    env.close()
    print('total avg:',score_total / (count - 1))
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.plot(score_lst)
    plt.title('Score in Test when stability = ', stability)
    plt.xlabel('Test Number')
    plt.ylabel('Score')
    plt.show()


if __name__ == '__main__':
    eval()