from stable_baselines3.common.env_checker import check_env
import wilson_maze_env
from gymnasium import make


env = make(id='WilsonMaze-v0', render_mode="text", size=7, timelimit=30, random_seed=42,
           variable_target=True, training_prompts_file='prompts/small_dataset/prompts.npz')

# It will check your custom environment and output additional warnings if needed
check_env(env)