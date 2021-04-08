# Reinforcement Learning
CM50270: Reinforcement Learning Project

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
