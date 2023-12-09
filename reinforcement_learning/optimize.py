from glass_bridge import GlassBridgeEnv
from experiments import run_episode
from q_learning import train_q_learning_agent
from sarsa import train_sarsa_agent
import numpy as np 
from tqdm import tqdm

def main():
    glass_bridge = GlassBridgeEnv()
    learning_episodes = 1000
    trial_episodes = 1000
    num_players = 6

    max_alpha = {}
    max_gamma = {}
    success_ratio = {}
    positions = {}

    policies = ['q_epsilon_greedy', 'q_random', 'q_softmax', 'q_posterior_sampling',
                   'sarsa_epsilon_greedy', 'sarsa_random', 'sarsa_softmax', 'sarsa_posterior_sampling', 'baseline']
    for policy in policies:
        max_alpha[policy] = 0
        max_gamma[policy] = 0
        success_ratio[policy] = 0
        positions[policy] = 0

    
    spacing = np.linspace(0, 1, 21)
    for alpha in tqdm(spacing, desc='alpha'):
        for gamma in tqdm(spacing, desc='gamma'):
            Q_epsilon_greedy = train_q_learning_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="epsilon-greedy")
            Q_explore = train_q_learning_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="all-random")
            Q_softmax = train_q_learning_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="softmax")
            Q_posterior_sampling = train_q_learning_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="posterior-sampling")
            SARSA_epsilon_greedy = train_sarsa_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="epsilon-greedy")
            SARSA_explore = train_sarsa_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="all-random")
            SARSA_softmax = train_sarsa_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="softmax")
            SARSA_posterior_sampling = train_sarsa_agent(glass_bridge, learning_episodes, alpha, gamma, exploration_strategy="posterior-sampling")
            
            for policy in policies:
                position_wins = {}
                for i in range(num_players):
                    position_wins[i] = 0
                total_reward = 0
                curr_pos = 0
                for ep in range(trial_episodes):
                    if policy == 'q_epsilon_greedy':
                        ep_reward = run_episode(glass_bridge, Q_epsilon_greedy, policy)
                    elif policy == 'q_random':
                        ep_reward = run_episode(glass_bridge, Q_explore, policy)
                    elif policy == 'q_softmax':
                        ep_reward = run_episode(glass_bridge, Q_softmax, policy)
                    elif policy == 'q_posterior_sampling':
                        ep_reward = run_episode(glass_bridge, Q_posterior_sampling, policy)
                    elif policy == 'sarsa_epsilon_greedy':
                        ep_reward = run_episode(glass_bridge, SARSA_epsilon_greedy, policy)
                    elif policy == 'sarsa_random':
                        ep_reward = run_episode(glass_bridge, SARSA_explore, policy)
                    elif policy == 'sarsa_softmax':
                        ep_reward = run_episode(glass_bridge, SARSA_softmax, policy)
                    elif policy == 'sarsa_posterior_sampling':
                        ep_reward = run_episode(glass_bridge, SARSA_posterior_sampling, policy)
                    elif policy == 'baseline':
                        ep_reward = run_episode(glass_bridge, SARSA_explore, policy)
                    
                    if ep_reward:
                        position_wins[curr_pos] += 1
                        curr_pos = 0
                    else:
                        if curr_pos >= num_players - 1: # no one won, start from player 0 again
                            curr_pos = 0
                        else:
                            curr_pos += 1
                    total_reward += ep_reward

                ratio = total_reward/trial_episodes*100
                if ratio > success_ratio[policy]:
                    success_ratio[policy] = ratio
                    max_alpha[policy] = alpha
                    max_gamma[policy] = gamma
                    positions[policy] = position_wins
    
    for policy in sorted(success_ratio, key=success_ratio.get, reverse=True):
        print('\n' + policy + ':')
        print('alpha: ' + str(max_alpha[policy]))
        print('gamma: ' + str(max_gamma[policy]))
        print('percentage of successful episodes: ' + str(success_ratio[policy]))
        print('position wins:', positions[policy])

    
if __name__ == '__main__':
    main()