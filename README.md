# Reinforcement Learning
CM50270: Reinforcement Learning Project

# RL Algorithms
![image](https://user-images.githubusercontent.com/366335/114033371-1c666400-9875-11eb-93fd-7cfb998ce0f8.png)

# Candidate Algorithms
- Model-Free
  - VPG
  - PPO
  - DDPG
  - SAC

# Setup
- Install Anaconda
  - https://docs.continuum.io/anaconda/install/linux
- Install OpenAI Spinning Up
  - https://spinningup.openai.com/en/latest/user/installation.html

# Environments
- MuJoCo(physics engine) - Paid: But free for students
  - http://www.mujoco.org/index.html
- OpenAI Gym - Free
  - Classic control
    - https://gym.openai.com/envs/#classic_control
  - Box2D
    - https://gym.openai.com/envs/#box2d

# Exporting Video
In your agent, import
```python
from gym import wrappers
from time import time # just to have timestamps in the files
```
Then, insert a wrapper call after you make the env.
```python
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
```
This will save .mp4 files to ./videos/1234/something.mp4

# Coursework
- Project Report(Overleaf Link)
  - https://www.overleaf.com/9711224264kfvktffyrctr
- video presentation of the project
- source code
- video of agent performance before and after learning

# Issue
- Rendering issue
  - https://stackoverflow.com/questions/60922076/pyglet-canvas-xlib-nosuchdisplayexception-cannot-connect-to-none-only-happens/67367292#67367292

# Reference
- David Silver
  - https://www.davidsilver.uk/teaching/
  - https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
  - https://www.youtube.com/watch?v=uPUEq8d73JI
- OpenAI Gym
  - https://spinningup.openai.com/en/latest/
  - https://openai.com/blog/spinning-up-in-deep-rl/
  - https://gym.openai.com/docs/#environments
  - https://gym.openai.com/envs/#box2d
  - Papers
    - https://arxiv.org/pdf/1606.01540.pdf
  - Old Contests
    - https://contest.openai.com/2018-1/
    - https://openai.com/blog/gym-retro/
- Grant Sanderson
  - https://www.youtube.com/watch?v=aircAruvnKk
- Deep RL Course
  - http://rail.eecs.berkeley.edu/deeprlcourse/
  - https://sites.google.com/view/deep-rl-bootcamp/lectures
- RL Unit Session
  - https://minerl.io/
  - https://www.youtube.com/watch?v=4ohomnzr1LM
- Frameworks
  - https://github.com/DLR-RM/stable-baselines3
