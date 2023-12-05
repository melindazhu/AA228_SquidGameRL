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

def softmax_exploration(Q, state, temperature):
    action_values = Q[state, :]
    action_probabilities = np.exp(action_values / temperature) / np.sum(np.exp(action_values / temperature))
    action = np.random.choice(len(action_probabilities), p=action_probabilities)
    return action


def posterior_sampling_exploration(Q, state, noise=0.1, num_samples=2):
    num_actions = Q.shape[1]

    # Draw samples from the posterior distribution for each action
    q_samples = np.random.normal(Q[state, :], noise, size=(num_samples, num_actions))
    # Assuming Q[state, :, 0] is mean and Q[state, :, 1] is standard deviation

    # Calculate the mean Q-value for each action across the samples
    mean_q_values = np.mean(q_samples, axis=0)

    # Choose the action with the highest mean Q-value
    action = np.argmax(mean_q_values)

    return action


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
        elif exploration_strategy == "softmax":
            a = softmax_exploration(Q, s, temperature=100)
        elif exploration_strategy == "posterior-sampling":
            a = posterior_sampling_exploration(Q, s, noise=0.1, num_samples=2)
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
    Q = train_q_learning_agent(glass_bridge, episodes, alpha, gamma, exploration_strategy="posterior-sampling")
    # normalized_Q = normalize(Q)
    # print(normalized_Q)

if __name__ == '__main__':
    main()