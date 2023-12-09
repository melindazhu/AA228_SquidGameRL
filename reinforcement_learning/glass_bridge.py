import gym
from gym.envs.toy_text import discrete
import numpy as np
import sys


ACTION_RIGHT = 0
ACTION_DIAGONAL = 1

MAP = {
    "5x2": ["SO--G", "--OO-"],
    "8x2": ["SO--O-O-", "--OO-O-G"],
    "10x2": ['--------O-', 'SOOOOOOO-G'],
    "12x2": ['--OO-O--OO-G', 'SO--O-OO--O-']
}

class GlassBridgeEnv(discrete.DiscreteEnv):
    """
    A GlassBridgeEnv object defines the state space of the MDP.
    The letter indicators are as follows: 
    S : starting point, safe
    - : thin glass, fail
    O : tempered glass, safe
    G : goal -> reward = 1 if reached

    Follows the class framework of OpenAI Gym environments.
    """
    def __init__(self):
        bridge_map = MAP["8x2"]
        self.bridge_map = bridge_map = np.asarray(bridge_map, dtype="c")
        self.n_rows, self.n_cols = n_rows, n_cols = bridge_map.shape
        self.possible_rewards = (0, 1)

        n_actions = 2
        n_states = n_rows * n_cols
        
        initialize_states = np.array(bridge_map == b"S").astype("float64").ravel()
        initialize_states /= initialize_states.sum()

        # define transition probabilities
        # T[s][a] is a list of tuples representing the possible transitions from state `s` with action `a`.
        # Each tuple contains (probability, s_prime, reward, done). 
        T = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

        # return the state (i.e. position on the bridge) specified by (row, col)
        def to_s(row, col):
            return row * n_cols + col

        # return the new (rol, col) after an action is performed
        def inc(row, col, action):
            col = min(col + 1, n_cols - 1)
            if action == ACTION_DIAGONAL:
                if row == 0:
                    row = 1
                elif row == 1:
                    row = 0
            return (row, col)

        # returns (newstate, reward, done)
        def update_probability_matrix(row, col, action):
            next_row, next_col = inc(row, col, action)
            s_prime = to_s(next_row, next_col)
            new_letter = bridge_map[next_row, next_col]
            done = bytes(new_letter) in b"G-"
            r = float(new_letter == b"G")
            return s_prime, r, done
        
        for row in range(n_rows):
            for col in range(n_cols):
                s = to_s(row, col)
                for a in range(n_actions): # only two possible actions
                    known_transitions = T[s][a]
                    curr_letter = bridge_map[row, col]
                    if curr_letter in b"G-": # reached the end of bridge
                        known_transitions.append((1.0, s, 0, True)) # (prob, s_prime, r, done)
                    else:
                        known_transitions.append((0.8, *update_probability_matrix(row, col, a)))
                        known_transitions.append((0.2, *update_probability_matrix(row, col, (a+3)%2)))
        # print(f"known transitions: {known_transitions}")
        super(GlassBridgeEnv, self).__init__(n_states, n_actions, T, initialize_states)

    def render(self):
        outfile = sys.stdout
        row, col = self.s // self.n_cols, self.s % self.n_cols
        bridge_map = self.bridge_map.tolist()
        bridge_map = [[c.decode("utf-8") for c in line] for line in bridge_map]
        bridge_map[row][col] = gym.utils.colorize(bridge_map[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Right", "Diagonal"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in bridge_map) + "\n")
    

# for testing only; this only gets run when we run glass_bridge.py as a standalone script
def main(): 
    # this works for one fixed-size grid, WILL CHANGE LATER (11/25 mz)
    env = GlassBridgeEnv()
    for i_episode in range(5): # number of trials we want to try the game
        newstate = env.reset()
        for t in range(10):
            env.render() # displays the current bridge environment and the current state
            # print(newstate)
            action = env.action_space.sample() # pick a random action
            newstate, reward, done, info = env.step(action)
            if done:
                if reward:
                    print("Crossed the bridge! Received reward of 1.")
                if not reward:
                    print("Oops! I've fallen!")
                print("Trial finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == "__main__":
    main()