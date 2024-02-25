import numpy as np

from stable_baselines3.common.env_checker import check_env
import wilson_maze_env
from gymnasium import make

prompts = np.random.randn(100, 300)

env = make(id='WilsonMaze-v0', render_mode="text", size=7, timelimit=30, random_seed=42, prompts=prompts, should_pickup_coins=True, 
           number_of_targets=4)

# It will check your custom environment and output additional warnings if needed
print(check_env(env))