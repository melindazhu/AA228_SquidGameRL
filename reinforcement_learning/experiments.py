"""
Compare the performance of different learning strategies.
Currently (11/26 mz):
    - Choosing random actions (100% exploration)
    - Choosing actions with highest Q-value (100% exploitation)
"""
from glass_bridge import GlassBridgeEnv
from q_learning import train_q_learning_agent, normalize
import numpy as np  

def run_episode(glass_bridge: GlassBridgeEnv(), Q: np.ndarray, policy: str) -> float:
    state = glass_bridge.reset()
    done = False
    total_return = 0
    while not done:
        if policy == 'random':
            action = glass_bridge.action_space.sample()
        elif policy == 'greedy_q':
            action = np.argmax(Q[state, :])
        else:
            print('Error: please select a policy from [random, greedy_q]')
            return
        s_prime, r, done, _ = glass_bridge.step(action)
        total_return += r
        state = s_prime
    return total_return


def main():
    trial_episodes = 1000
    glass_bridge = GlassBridgeEnv()
    alpha = 0.1
    gamma = 0.95
    q_learning_episodes = 2000
    Q = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma)
    for policy in ['random', 'greedy_q']:
        total_reward = 0
        for ep in range(trial_episodes):
            total_reward += run_episode(glass_bridge, Q, policy)
        print(f"Percentage of successful episodes for policy {policy}: {total_reward / trial_episodes * 100}%")
    
if __name__ == '__main__':
    main()
