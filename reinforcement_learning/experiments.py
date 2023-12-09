"""
Compare the performance of different learning strategies.
Currently (11/26 mz):
    - Choosing random actions (100% exploration)
    - Choosing actions with highest Q-value (100% exploitation)
"""
from glass_bridge import GlassBridgeEnv
from q_learning import train_q_learning_agent, normalize
from sarsa import train_sarsa_agent
import numpy as np  

"""
Only run one episode. The episode exits when someone falls or someone makes it to the end.
"""
def run_episode(glass_bridge: GlassBridgeEnv(), Q: np.ndarray, policy: str) -> float:
    state = glass_bridge.reset()
    done = False
    total_return = 0
    while not done:
        if 'baseline' in policy:
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
    alpha = 0.9
    gamma = 0.35
    q_learning_episodes = 1000
    Q_epsilon_greedy = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma, exploration_strategy="epsilon-greedy")

    # Experiment 3: Q-learning w/ softmax exploration
    alpha = 0.9
    gamma = 0.2
    q_learning_episodes = 1000
    Q_softmax = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma, exploration_strategy="softmax")

    # Experiment 4: Q-learning w/ posterior sampling exploration
    alpha = 0.4
    gamma = 0.25
    q_learning_episodes = 1000
    Q_posterior_sampling = train_q_learning_agent(glass_bridge, q_learning_episodes, alpha, gamma, exploration_strategy="posterior-sampling")

    # Experiment 5: Sarsa with epsilon greedy
    glass_bridge = GlassBridgeEnv()
    alpha = 0.35
    gamma = 0.7
    sarsa_learning_episodes = 1000
    sarsa_epsilon_greedy = train_sarsa_agent(glass_bridge, sarsa_learning_episodes, alpha, gamma, exploration_strategy="epsilon-greedy")

    # Experiment 7: Sarsa w/ softmax exploration
    alpha = 0.1
    gamma = 0.85
    sarsa_learning_episodes = 1000
    sarsa_softmax = train_sarsa_agent(glass_bridge, sarsa_learning_episodes, alpha, gamma, exploration_strategy="softmax")

    # Experiment 8: Sarsa w/ posterior sampling exploration
    alpha = 0.4
    gamma = 1.0
    sarsa_learning_episodes = 1000
    sarsa_posterior_sampling = train_sarsa_agent(glass_bridge, sarsa_learning_episodes, alpha, gamma, exploration_strategy="posterior-sampling")

    trial_episodes = 1000

    for policy in ['q_epsilon_greedy', 'q_baseline', 'q_posterior_sampling', 'q_softmax', 'sarsa_epsilon_greedy', 'sarsa_softmax', 'sarsa_posterior_sampling']:
        position_wins = {}
        for i in range(6):
            position_wins[i] = 0
        print(f"Running with policy {policy}")
        total_reward = 0
        curr_pos = 0
        for ep in range(trial_episodes):
            if policy == 'q_epsilon_greedy':
                ep_reward = run_episode(glass_bridge, Q_epsilon_greedy, policy)
            elif policy == 'q_softmax':
                ep_reward = run_episode(glass_bridge, Q_softmax, policy)
            elif policy == 'q_posterior_sampling':
                ep_reward = run_episode(glass_bridge, Q_posterior_sampling, policy)
            elif policy == 'q_baseline': # dw about Q, we don't use it in run_episode
                ep_reward = run_episode(glass_bridge, Q_softmax, policy)
            elif policy == 'sarsa_epsilon_greedy':
                ep_reward = run_episode(glass_bridge, sarsa_epsilon_greedy, policy)
            elif policy == 'sarsa_softmax':
                ep_reward = run_episode(glass_bridge, sarsa_softmax, policy)
            elif policy == 'sarsa_posterior_sampling':
                ep_reward = run_episode(glass_bridge, sarsa_posterior_sampling, policy)
            
            if ep_reward:
                position_wins[curr_pos] += 1
                curr_pos = 0
            else:
                if curr_pos >= 5: # no one won, start from player 0 again
                    curr_pos = 0
                else:
                    curr_pos += 1
            total_reward += ep_reward
        print(f"Percentage of successful episodes for policy {policy}: {total_reward / trial_episodes * 100}%")
        print(f"Successes per player position for policy {policy}:")
        for j in range(6):
            print(f"Wins for player {j}: {position_wins[j]}")
    
if __name__ == '__main__':
    main()
