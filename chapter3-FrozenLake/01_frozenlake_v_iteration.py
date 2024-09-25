"""
rl训练
Author: yangjie
Date: 2024-09-23
"""

import random
from env import Env
import collections
from tensorboardX import SummaryWriter

GAMMA = 0.9
TEST_EPISODES = 20


# 定义Agent类，该类包括上述表以及在训练循环中用到的函数
class Agent:
    def __init__(self):
        self.env = Env()
        self.state = 0
        # 定义奖励表、转移表、价值表
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """
        从环境中收集随机经验，并更新奖励表和转移表。
        无须等片段结束就可以开始学习。
        价值迭代法和交叉熵方法的区别之一。交叉熵方法只在完整的片段中学习。
        """
        for _ in range(count):
            action = random.sample([0, 1, 2, 3], 1)[0]
            new_state, reward, is_done, _, _ = self.env.step(action)
            new_state = new_state.index(1)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state = 0
            else:
                self.state = new_state

    def calc_action_value(self, state, action):
        """
        根据转移表、奖励表和价值表计算从状态采取某动作的价值
        针对某状态选择最佳动作，并在价值迭代时计算状态的新价值，主要执行以下步骤进行价值估计：
            1.转移表中获取给定状态和动作的转移计数器
            2.对动作所到达的每个目标状态进行迭代，并使用Bellman方程计算其对总动作价值的贡献，它也等于立即奖励和折扣长期状态价值之和。
        """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state):
        """
        决定某状态可采取的最佳动作
        这个动作选择过程是确定性的，因为play_n_random_steps()函数引入了足够的探索
        """
        best_action, best_value = None, None
        for action in range(4):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
        测试bellman方程有效性：使用select_action()来查找要采取的最佳动作，并在环境中运行一整个片段
        """
        total_reward = 0.0
        state = 0
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _, _ = env.step(action)
            new_state = new_state.index(1)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        为每个该状态可到达的状态计算价值，从而获得状态价值的候选项。
        用状态可执行动作的最大价值作为当前状态的价值。
        """
        for state in range(16):
            state_values = [
                self.calc_action_value(state, action) for action in range(4)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = Env()
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        print(reward, best_reward)
        if reward > best_reward:
            print(f"Best reward updated {best_reward} -> {reward}")
            best_reward = reward
        if reward > 0.90:
            print(f"Solved in {iter_no} iterations!")
            break
    writer.close()
