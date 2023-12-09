from glass_bridge import GlassBridgeEnv
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
from typing import Optional
from q_learning import softmax_exploration, posterior_sampling_exploration, normalize


def choose_action(
    glass_bridge: GlassBridgeEnv(),
    Q: np.ndarray,
    s: int,
    exploration_strategy: str,
    epsilon: Optional[float] = 1.0
) -> int:
    a = None
    if exploration_strategy == "epsilon-greedy":
        if np.random.random() < epsilon:
            a = glass_bridge.action_space.sample()
        else:
            a = np.argmax(Q[s,:])
    elif exploration_strategy == "softmax":
        a = softmax_exploration(Q, s, temperature=100)
    elif exploration_strategy == "posterior-sampling":
        a = posterior_sampling_exploration(Q, s, noise=0.1, num_samples=2)
    else:
        a = glass_bridge.action_space.sample()
    return a
    

def run_sarsa(
    glass_bridge: GlassBridgeEnv(),
    Q: np.ndarray,
    alpha: float,
    gamma: float,
    exploration_strategy: str,
    epsilon: Optional[float] = 1.0
) -> np.ndarray:
    s = glass_bridge.reset()
    a = choose_action(glass_bridge, Q, s, exploration_strategy, epsilon)
    done = False
    while not done:
        s_prime, r, done, _ = glass_bridge.step(a)
        a_prime = glass_bridge.action_space.sample()
        Q[s, a] += alpha * (r + gamma * Q[s_prime, a_prime]- Q[s, a])
        s = s_prime
        a = a_prime
    return Q


def train_sarsa_agent(
    glass_bridge: GlassBridgeEnv(),
    episodes: int,
    alpha: float,
    gamma: float,
    exploration_strategy: str
) -> np.ndarray:
    Q = np.zeros([glass_bridge.observation_space.n, glass_bridge.action_space.n])
    epsilon = 1.0
    for i in range(episodes):
        Q = run_sarsa(glass_bridge, Q, alpha, gamma, exploration_strategy, epsilon)
        if epsilon > 0.05:
            epsilon *= 0.98
        clear_output(wait=True)
    return Q


def main():
    alpha = 0.1
    gamma = 0.95
    episodes = 1000
    glass_bridge = GlassBridgeEnv()
    Q = train_sarsa_agent(glass_bridge, episodes, alpha, gamma, exploration_strategy="posterior-sampling")
    normalized_Q = normalize(Q)
    print(normalized_Q)

if __name__ == '__main__':
    main()