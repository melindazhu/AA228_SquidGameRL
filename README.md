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

## Part 1: Creating an RL agent <br>
Formulate the problem as a Markov Decision Process: 
- **agent** = player
- **environment** = the glass bridge, defined as a grid
- **reward** = 1 if whole bridge is traversed, 0 otherwise
- **action** = moving to next adjacent position or the far diagonal position
- **state** = position of the agent on the glass bridge grid