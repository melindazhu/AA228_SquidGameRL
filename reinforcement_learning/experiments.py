"""
Compare the performance of different learning strategies.
Currently (11/26 mz):
    - Choosing random actions (100% exploration)
    - Choosing actions with highest Q-value (100% exploitation)
"""
from glass_bridge import GlassBridgeEnv
from q_learning import train_q_learning_agent, normalize
import numpy as np  

"""
Only run one episode. The episode exits when someone falls or someone makes it to the end.
"""
def run_episode(glass_bridge: GlassBridgeEnv(), Q: np.ndarray, policy: str) -> float:
    state = glass_bridge.reset()
    done = False
    total_return = 0
    while not done:
        if policy == 'baseline':
            action = glass_bridge.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        s_prime, r, done, _ = glass_bridge.step(action)
        total_return += r
        state = s_prime
    return total_return


def main():
    # Experiment 1: Q-learning with epsilon greedy
    glass_bridge = GlassBridgeEnv()
    alpha = 0.05
    gamma = 0.95
    q_learning_episodes = 20000
    Q_epsilon_greedy = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma, exploration_strategy="epsilon-greedy")

    # Experiment 2: Q-learning w/ 100% random exploration 
    alpha = 0.1
    gamma = 0.95
    q_learning_episodes = 2000
    Q_explore = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma, exploration_strategy="all-random")

    # initialize number of wins per player position
    trial_episodes = 1020
    position_wins = {}
    for i in range(6):
        position_wins[i] = 0

    for policy in ['q_epsilon_greedy', 'q_random', 'baseline']:
        position_wins = {}
        for i in range(6):
            position_wins[i] = 0
        print(f"Running with policy {policy}")
        total_reward = 0
        for ep in range(trial_episodes):
            if policy == 'q_epsilon_greedy':
                ep_reward = run_episode(glass_bridge, Q_epsilon_greedy, policy)
            elif policy == 'q_random':
                ep_reward = run_episode(glass_bridge, Q_explore, policy)
            elif policy == 'baseline': # dw about Q, we don't use it in run_episode
                ep_reward = run_episode(glass_bridge, Q_explore, policy)
            
            if ep_reward:
                position_wins[ep % 6] += 1
            total_reward += ep_reward
        print(f"Percentage of successful episodes for policy {policy}: {total_reward / trial_episodes * 100}%")
        print(f"Successes per player position for policy {policy}:")
        for j in range(6):
            print(f"Wins for player {j}: {position_wins[j]}")
    
if __name__ == '__main__':
    main()
