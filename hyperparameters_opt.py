import json
import os
import sys
import numpy as np
import joblib

from typing import Any, Dict
from pprint import pprint
from copy import deepcopy
import warnings

import optuna
import optuna_distributed
from optuna.trial import TrialState

from torch import nn
from sklearn.model_selection import train_test_split

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from common import get_npz_input_data, make_env
from eval_model import evaluate_model


STUDY_NAME = "ppo-distributed"
STORAGE = "mysql://root@localhost/example"

PROMPTS_FILE = "prompts/big_dataset/embeds/llama/llama-2/llama-2-7b-f16.npz"
TARGETS_FILE = "prompts/big_dataset/prompts.csv"

RANDOM_STATE = 42

class WilsonMazeEvalCallback(BaseCallback):
    def __init__(self, eval_config, trial, best_model_save_path=None, logs_path=None, 
                 eval_freq=10000, deterministic=True, max_number_of_steps=15, 
                 use_action_masks=False, verbose=0, kwargs=None):
        super(WilsonMazeEvalCallback, self).__init__(verbose)

        self.eval_config = eval_config
        self.kwargs = kwargs

        self.best_score = -np.inf
        self.last_eval_score = -np.inf
        self.best_model_save_path = best_model_save_path
        self.logs_path = logs_path
        self.verbose = verbose

        self.use_action_masks = use_action_masks
        self.max_number_of_steps = max_number_of_steps
        self.eval_freq = eval_freq
        self.deterministic = deterministic

        self.trial = trial
        self.is_pruned = False
        self.eval_idx = 0
    
    def _init_callback(self) -> None:
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        if self.logs_path is not None:
            os.makedirs(self.logs_path, exist_ok=True)
            if not os.path.isfile(self.logs_path + '/evaluations.json'):
                with open(self.logs_path + '/evaluations.json', 'w') as log_file:
                    j_kwarags = deepcopy(self.kwargs)
                    j_kwarags["policy_kwargs"]['activation_fn'] = j_kwarags["policy_kwargs"]['activation_fn'].__name__
                    json.dump({'evaluations': [], 'kwargs': j_kwarags}, log_file, indent=4)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1

            eval_full_score, eval_partial_score, stats = evaluate_model(self.model, self.training_env,
                                            deterministic=self.deterministic,
                                            use_action_masks=self.use_action_masks,
                                            max_number_of_steps=self.max_number_of_steps,
                                            verbose=self.verbose > 1,
                                            **self.eval_config)
            self.last_eval_score = max(eval_full_score, eval_partial_score)

            if self.logs_path is not None:
                with open(os.path.join(self.logs_path, 'evaluations.json'), 'r+') as log_file:
                    old_data = json.load(log_file)
                    old_data['evaluations'].append({'step': self.num_timesteps,
                                                    'eval_score': self.last_eval_score,
                                                    'stats': stats})
                    log_file.seek(0)
                    json.dump(old_data, log_file, indent=4)
                    log_file.truncate()

            if self.verbose:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_score={self.last_eval_score:.2f}")

            self.logger.record("eval/eval_score", self.last_eval_score)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if self.last_eval_score > self.best_score:
                self.best_score = self.last_eval_score

                if self.verbose >= 1:
                    print("New best score!")
                if self.best_model_save_path is not None:
                    vec_normalizer = self.model.get_vec_normalize_env()
                    if vec_normalizer:
                        vec_normalizer.save(os.path.join(self.best_model_save_path, "best_model_vec_normalizer.pkl"))
                    self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))
                
            
            self.trial.report(self.last_eval_score, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
            
        return True

class Study:
    def __init__(self, n_timesteps=50000, n_evaluations=4, n_envs=8, verbose=0, logs_path='./logs/trials', 
                 eval_max_number_of_steps=15, use_action_maks=False) -> None:
        
        self.X_train, self.X_valid, self.y_train, self.y_valid = None, None, None, None
        self.n_timesteps = n_timesteps
        self.n_evaluations = n_evaluations
        self.n_envs = n_envs
        self.verbose = verbose
        self.logs_path = logs_path
        self.eval_max_number_of_steps = eval_max_number_of_steps
        self.use_action_masks = use_action_maks

    def set_data(self, prompts_file: str, targets_file: str, 
                 limit: int = 6000, test_size=0.2, seed: int = RANDOM_STATE):
        """
            Set the input data for the study.

            :param prompts_file: the path to the prompts file
            :param targets_file: the path to the targets file
            :param limit: the limit of the input data
            :param test_size: the size of the validation set
            :param seed: the random seed for the data split
        """
        assert limit > 1, "Limit must be greater than 0"

        X, Y = get_npz_input_data(prompts_file, targets_file)
        X, Y = X[:limit], Y[:limit]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, Y, test_size=test_size, 
                                                                random_state=seed, stratify=Y[:, 0])
    
    def sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sampler for PPO hyperparams.

        :param trial:
        :return:
        """
        # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        # n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        gamma = trial.suggest_categorical("gamma", [0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        # learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        ent_coef = trial.suggest_categorical("ent_coef", [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
        # clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        # gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_categorical("vf_coef", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu'])
        # lr_schedule = "constant"
        # Uncomment to enable learning rate schedule
        # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
        # if lr_schedule == "linear":
        #     learning_rate = linear_schedule(learning_rate)

        # TODO: account when using multiple envs
        # if batch_size > n_steps:
        #     batch_size = n_steps

        # Independent networks usually work best
        # when not working with images
        net_arch = {
            "tiny": dict(pi=[64], vf=[64]),
            "small": dict(pi=[128, 64], vf=[128, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }[net_arch_type]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]

        return {
            "n_steps": 2 ** 11,
            "batch_size": 2 ** 6,
            "gamma": gamma,
            # "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            # "clip_range": clip_range,
            # "gae_lambda": gae_lambda,
            # "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "policy_kwargs": dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=False,
            ),
        }

    def objective(self, trial: optuna.Trial) -> float:
        self.env_train_config = {
            "env_id": "WilsonMaze-v0",
            "render_mode": "text",
            "size": 7,
            "timelimit": 30,
            "random_seed": 42,
            "variable_target": False,
            "add_coins": True,
            "terminate_on_wrong_target": True,
            "prompt_size": 1024,
            "prompt_mean": True
        }

        self.env_eval_config = deepcopy(self.env_train_config)
        self.env_eval_config["timelimit"] = 5
        self.env_eval_config["prompts"] = self.X_valid
        self.env_eval_config["labels"] = self.y_valid
        self.env_eval_config["id"] = self.env_eval_config["env_id"]
        del self.env_eval_config["env_id"]

        # Pass n_actions to initialize DDPG/TD3 noise sampler
        # Sample candidate hyperparameters
        sampled_hyperparams = self.sample_ppo_params(trial)

        vec_env = DummyVecEnv(
        [make_env(rank=i, x=self.X_train, y=self.y_train, seed=0, **self.env_train_config)
         for i in range(self.n_envs)])
        vec_env = VecNormalize(vec_env, norm_reward=False)

        if self.use_action_masks:
            model = MaskablePPO("MlpPolicy", vec_env, device='auto', tensorboard_log=None, 
                                seed=RANDOM_STATE, verbose=self.verbose > 1, **sampled_hyperparams)
        else:
            model = PPO("MlpPolicy", vec_env, device='auto', tensorboard_log=None, 
                        seed=RANDOM_STATE, verbose=self.verbose > 1, **sampled_hyperparams)

        eval_freq = max(int(self.n_timesteps // self.n_evaluations // self.n_envs), 1)

        path = None
        if self.logs_path is not None:
            path = os.path.join(self.logs_path, f"trial_{trial.number!s}")
    
        eval_callback = WilsonMazeEvalCallback(
            self.env_eval_config,
            trial,
            best_model_save_path=path,
            logs_path=path,
            eval_freq=eval_freq,
            deterministic=True,
            max_number_of_steps=self.eval_max_number_of_steps,
            verbose=self.verbose,
            kwargs=sampled_hyperparams
        )

        try:
            model.learn(self.n_timesteps, callback=eval_callback)  # type: ignore[arg-type]
            # Free memory
            assert model.env is not None
            model.env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            assert model.env is not None
            model.env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_eval_score

        del model.env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward
    
    # def prune_study(study_path):
    #     """
    #         https://medium.com/@vojtechmolek/speeding-up-optuna-sampling-and-objective-computation-1fbb179b5edd
    #     """
    #     counter = 0
    #     study = joblib.load(study_path)
    #     study = study._study # get vanilla Optuna study from DistributedStudy

    #     for trial in study.get_trials(deepcopy=False): # get all trials from study
    #         if trial
    #         trial.state=optuna.trial.TrialState.COMPLETE
    #         counter += 1
    #         else:
    #         trial.state=optuna.trial.TrialState.FAIL

    #     dist_study=opt_dst.from_study(study) # create new DistributedStudy
    #     print(f"Number of trials used for sampling {counter}")
    #     joblib.dump(dist_study, study_path)

    def optimize(self, n_trials: int, timeout: float = None, n_startup_trials: int = 10, seed: int = RANDOM_STATE):
        """
            Optimize the study using Optuna.

            :param n_trials: the number of trials to run
            :param timeout: the timeout for the study
        """
        if os.path.isfile(f'{STUDY_NAME}_study.pkl'):
            warnings.warn("Study already exists, loading it from file")
            dist_study = joblib.load(f'{STUDY_NAME}_study.pkl')
            n_trials = n_trials - len(study.trials)
        else:
            sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=True)
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4)

            study = optuna.create_study(
                study_name=STUDY_NAME,
                direction="maximize",
                # storage=STORAGE,
                sampler=sampler,
                pruner=pruner
            )
            dist_study = optuna_distributed.from_study(study)

        dist_study.optimize(self.objective, 
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True, n_jobs=-1)
        
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("Best Value: ", trial.value)

        print("Best Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        joblib.dump(dist_study, f'{STUDY_NAME}_study.pkl')
        

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else RANDOM_STATE

    study = Study(n_timesteps=1e6, n_evaluations=4, n_envs=8, verbose=1, logs_path='./logs/trials', use_action_maks=False)
    study.set_data(PROMPTS_FILE, TARGETS_FILE, limit=4960, test_size=0.2, seed=RANDOM_STATE)

    study.optimize(n_trials=100, timeout=18000, n_startup_trials=5, seed=seed + RANDOM_STATE)
