"""
Currently, this Q-learning agent chooses the actions with the highest 
Q-value in order to maximize the reward; i.e. a greedy policy. 
This is different from the random policy which chooses completely 
random actions and utilizes zero exploitation characteristics.
"""
from glass_bridge import GlassBridgeEnv
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
from typing import Optional

def bellman_update(
    Q: np.ndarray,
    alpha: float,
    gamma: float,
    r: float,
    s: int,
    s_prime: int,
    a: int
) -> np.ndarray:
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_prime,:]) - Q[s, a])
    return Q


def run_q_learning(
    glass_bridge: GlassBridgeEnv(),
    Q: np.ndarray,
    alpha: float,
    gamma: float,
    exploration_strategy: str,
    epsilon: Optional[float] = 1.0
) -> np.ndarray:
    s = glass_bridge.reset()
    done = False
    while not done:
        if exploration_strategy == "epsilon-greedy":
            if np.random.random() < epsilon:
                a = glass_bridge.action_space.sample()
            else:
                a = np.argmax(Q[s,:])
        else: # "q-random"
            a = glass_bridge.action_space.sample() # pick completely random action
        s_prime, r, done, _ = glass_bridge.step(a)
        Q = bellman_update(Q, alpha, gamma, r, s, s_prime, a)
        s = s_prime
    return Q


def train_q_learning_agent(
    glass_bridge: GlassBridgeEnv(),
    episodes: int,
    alpha: float,
    gamma: float,
    exploration_strategy: str
) -> np.ndarray:
    Q = np.zeros([glass_bridge.observation_space.n, glass_bridge.action_space.n])
    ep_list = range(episodes)
    epsilon = 1.0
    for i in tqdm(ep_list, desc="Running Q-learning episodes"):
        Q = run_q_learning(glass_bridge, Q, alpha, gamma, exploration_strategy, epsilon)
        if epsilon > 0.05:
            epsilon *= 0.98
        clear_output(wait=True)
    return Q


def normalize(Q) -> np.ndarray:
    Q_max = np.max(Q)
    if Q_max > 0.0:
        Q = (Q / Q_max)*1
    return Q


# for testing only; this only gets run when we run glass_bridge.py as a standalone script
def main():
    alpha = 0.1
    gamma = 0.95
    episodes = 1000
    glass_bridge = GlassBridgeEnv()
    Q = train_q_learning_agent(glass_bridge, episodes, alpha, gamma, exploration_strategy="epsilon-greedy")
    # normalized_Q = normalize(Q)
    # print(normalized_Q)

if __name__ == '__main__':
    main()