# AA228 Final Project - Squid Game Glass Bridge w/ RL & Probabilistic Reasoning

## Prerequisites
1. Clone this repository: `$ git clone https://github.com/melindazhu/AA228_SquidGameRL.git`
2. Set up a Python virtual environment. We do this because package versions were specifically compatible with the packages listed in requirements.txt. Run the following: 
    - `$ pip install virtualenv` in home directory
    - `$ cd /path/to/this/repo`
    - `$ virtualenv venv`
    - `$ source venv/bin/activate` for Mac and `venv\Scripts\activate` for Windows
3. Install packages required for this repo (WIP): `$ pip install -r requirements.txt`

## Make changes for your own testing, i.e. a crash course on Git/GitHub
TLDR about Git:
- We want to keep a "final" version on this branch of the code called `main`
- If you want to make local changes for your own testing but don't want everyone to see it yet, run the following:
    - `$ git checkout -b my_branch` # this creates your own local branch that you can work on.
    - make changes to the code
    - `$ git add file1 file2 etc.` # this stages files for a commit
    - `$ git commit -m "some message describing your change"` # this commits the changes to your local branch, essentially locking down the modifications you just made, and any changes made from this point on will be marked as new changes on top of your commit.
    - `$ git push` # this allows you to view changes on GitHub and create a pull request.
    - Create a pull request on GitHub, if you want to merge the changes into the `main` branch.

## Training an RL agent <br>
Formulate the problem as a Markov Decision Process: 
- **agent** = player
- **environment** = the glass bridge, defined as a grid
- **reward** = 1 if whole bridge is traversed, 0 otherwise
- **action** = moving to next adjacent position or the far diagonal position
- **state** = position of the agent on the glass bridge grid

This problem demonstrates the Markov property; i.e. the probability of receiving reward only depends on the current state, not on the previous sequence of states.

### Understanding the goal
We can view the glass bridge as a series of `n_rows_of_tiles` experiments in which each failure means that a player has fallen. Zero failures in a set of episodes means that the first player has made it across the bridge, while 1 failure means that the second player made it, etc.

In our problem setup, we'll say that there are `N` rows of tiles and `N-2` players, just like in Squid Game. The following is a high-level overview of our project workflow: 
1. Design an RL glass bridge environment that keeps track of states and actions, and simulates taking steps toward the end goal.
2. Train an RL agent to obtain the best `Q`, using different exploration-exploitation strategies with Q-learning and SARSA learning.
3. Use the learned `Q` to run the glass bridge several times, and calculate the frequency for each position in line's chance of winning.

### What we've done, and aim to do
- [x] Create an experimental Numpy-based visual environment that simulates players crossing a bridge, with any bridge dimensions. 
- [ ] Gather metrics on which player position wins the most frequently, using this experimental environment with different Nx2 sized grids.
- [x] Create an RL environment (using OpenAI Gym framework) that follows the defined MDP to run Q-learning, SARSA, etc.
- [x] Implement 3 exploration strategies and incorporate them into Q-learning.
    - [x] epsilon-greedy exploration
    - [x] softmax exploration
    - [x] posterior sampling exploration
- [ ] Implement an additional learning algorithm (for the 4-unit extension).
- [x] Train and report results for each RL strategy. (WiP, pending additional learning algorithm, but I wrote the code framework to record results.)
- [ ] Refine results using hyperparameter optimization. Interesting things to explore might be the discount factor, epsilon decay rate (to change the rate of exploration vs. exploitation), or learning rate.
- [ ] Report results for different Nx2 sized grids, and compare with results from item (2). Current default in the Results Tracker is for an 8x2 grid.

### Results Tracker
```
Running with policy q_epsilon_greedy
Percentage of successful episodes for policy q_epsilon_greedy: 22.84313725490196%
Successes per player position for policy q_epsilon_greedy:
Wins for player 0: 29
Wins for player 1: 40
Wins for player 2: 42
Wins for player 3: 38
Wins for player 4: 45
Wins for player 5: 39

Running with policy q_random
Percentage of successful episodes for policy q_random: 20.294117647058822%
Successes per player position for policy q_random:
Wins for player 0: 45
Wins for player 1: 28
Wins for player 2: 33
Wins for player 3: 45
Wins for player 4: 26
Wins for player 5: 30

Running with policy softmax
Percentage of successful episodes for policy softmax: 20.19607843137255%
Successes per player position for policy softmax:
Wins for player 0: 35
Wins for player 1: 34
Wins for player 2: 39
Wins for player 3: 32
Wins for player 4: 25
Wins for player 5: 41

Running with policy posterior-sampling
Percentage of successful episodes for policy posterior-sampling: 19.80392156862745%
Successes per player position for policy posterior-sampling:
Wins for player 0: 35
Wins for player 1: 31
Wins for player 2: 37
Wins for player 3: 33
Wins for player 4: 39
Wins for player 5: 27

Running with policy baseline
Percentage of successful episodes for policy baseline: 0.6862745098039216%
Successes per player position for policy baseline:
Wins for player 0: 0
Wins for player 1: 1
Wins for player 2: 0
Wins for player 3: 1
Wins for player 4: 3
Wins for player 5: 2
```