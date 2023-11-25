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

def run_q_learning(
    glass_bridge: GlassBridgeEnv(),
    Q: np.ndarray,
    alpha: float,
    gamma: float
) -> np.ndarray:
    s = glass_bridge.reset()
    done = False
    while not done:
        a = glass_bridge.action_space.sample() # pick random action
        s_prime, r, done, _ = glass_bridge.step(a)
        Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime, :]) - Q[s, a])
        s = s_prime
    return Q


def train_q_learning_agent(
    glass_bridge: GlassBridgeEnv(),
    episodes: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    Q = np.zeros([glass_bridge.observation_space.n, glass_bridge.action_space.n])
    ep_list = range(episodes)
    for i in tqdm(ep_list, desc="Running Q-learning episodes"):
        Q = run_q_learning(glass_bridge, Q, alpha, gamma)
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
    Q = train_q_learning_agent(glass_bridge, episodes, alpha, gamma)
    normalized_Q = normalize(Q)
    print(normalized_Q)

if __name__ == '__main__':
    main()