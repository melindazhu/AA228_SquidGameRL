# AA228 Final Project - Squid Game Glass Bridge w/ RL & Probabilistic Reasoning

## Prerequisites
1. Clone this repository: `$ git clone https://github.com/melindazhu/AA228_SquidGameRL.git`
2. Set up a Python virtual environment. We do this because package versions were specifically compatible with the packages listed in requirements.txt. Run the following: 
    - `$ pip install virtualenv` in home directory
    - `$ cd /path/to/this/repo`
    - `$ virtualenv venv`
    - `$ source venv/bin/activate` for Mac and `venv\Scripts\activate` for Windows
3. Install packages required for this repo (WIP): `$ pip install -r requirements.txt`


## Part 1: Creating an RL agent <br>
Formulate the problem as a Markov Decision Process: 
- **agent** = player
- **environment** = the glass bridge, defined as a grid
- **reward** = 1 if whole bridge is traversed, 0 otherwise
- **action** = moving to next adjacent position or the far diagonal position
- **state** = position of the agent on the glass bridge grid