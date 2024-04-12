import json
import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

from stable_baselines3.common.vec_env import VecNormalize

from eval_model import evaluate_model


class WilsonMazeCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0,
                 record_bad_behaviour=False,
                 record_coins_behaviour=False,
                 record_targets_dist=False,
                 record_targets_reset_dist=False,
                 record_prompts_dist=False,
                 evaluate=False,
                 eval_config=None,
                 eval_freq=10000,
                 best_model_save_path='./',
                 logs_path='./',
                 deterministic=False,
                 use_action_masks=False,
                 max_number_of_steps=10,
                 device='cpu'
                 ):
        super(WilsonMazeCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.move_wins = 0
        self.coin_wins = 0
        self.move_loses = 0
        self.coin_loses = 0
        self.endings = 0
        self.steps_to_end = 0
        self.steps_to_end = []
        self.out_of_bounds = 0
        self.wall_collisions = 0
        self.good_coins_behaviour = 0
        self.bad_coins_behaviour = 0

        self.targets = defaultdict(int)
        self.targets_per_envs = None
        self.reset_targets = None
        self.rollout_targets = defaultdict(int)
        self.prompts = defaultdict(lambda: defaultdict(int))
        self.rollout_prompts = defaultdict(lambda: defaultdict(int))
        self.best_eval_score = 0.0

        self.record_targets_dist = record_targets_dist
        self.record_targets_reset_dist = record_targets_reset_dist
        self.record_prompts_dist = record_prompts_dist
        self.record_bad_behaviour = record_bad_behaviour
        self.record_coins_behaviour = record_coins_behaviour
        self.evaluate = evaluate
        self.eval_freq = eval_freq
        self.eval_config = eval_config
        self.best_model_save_path = best_model_save_path
        self.logs_path = logs_path
        self.deterministic = deterministic
        self.use_action_masks = use_action_masks
        self.max_number_of_steps = max_number_of_steps
        self.device = device

        assert not self.evaluate or self.eval_config is not None, 'No eval config provided'

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.record_targets_dist:
            self.targets_per_envs = {k: defaultdict(int) for k in range(self.training_env.num_envs)}
        if self.record_targets_reset_dist:
            self.reset_targets = {k: defaultdict(int) for k in range(self.training_env.num_envs)}

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.logs_path is not None:
            os.makedirs(self.logs_path, exist_ok=True)
            if not os.path.isfile(self.logs_path + '/evaluations.json'):
                with open(self.logs_path + '/evaluations.json', 'w') as log_file:
                    json.dump({'evaluations': []}, log_file, indent=4)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.move_wins = 0
        self.coin_wins = 0
        self.move_loses = 0
        self.coin_loses = 0
        self.endings = 0
        self.steps = [0 for _ in range(self.training_env.num_envs)]
        self.steps_to_end = {i: [] for i in range(self.training_env.num_envs)}

        if self.record_bad_behaviour:
            self.out_of_bounds = 0
            self.wall_collisions = 0
        
        if self.record_coins_behaviour:
            self.good_coins_behaviour = 0
            self.bad_coins_behaviour = 0

        if self.record_targets_dist:
            self.rollout_targets.clear()

        if self.record_prompts_dist:
            self.rollout_prompts.clear()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        for i in range(self.training_env.num_envs):
            self.steps[i] += 1
            if self.locals['dones'][i]:
                if not self.locals['infos'][i]['TimeLimit.truncated']:
                    self.move_wins += 1
                    self.steps_to_end[i].append(self.steps[i])
                    self.steps[i] = 0
                    
                    if self.locals['rewards'][i] >= 2.0:
                        self.coin_wins += 1
                    else:
                        self.coin_loses += 1
                else:
                    self.move_loses += 1
                    self.coin_loses += 1
                self.endings += 1

                if self.record_targets_reset_dist:
                    self.reset_targets[i][self.locals['infos'][i]['target']] += 1

                if self.record_bad_behaviour:
                    self.out_of_bounds += self.locals['infos'][i]['out_of_bounds']
                    self.wall_collisions += self.locals['infos'][i]['wall_collisions']
                
                if self.record_coins_behaviour:
                    self.good_coins_behaviour += self.locals['infos'][i]['good_pickup_coins']
                    self.bad_coins_behaviour += self.locals['infos'][i]['bad_pickup_coins']

                if self.record_targets_dist:
                    self.rollout_targets[self.locals['infos'][i]['target']] += 1
                    self.targets_per_envs[i][self.locals['infos'][i]['target']] += 1

                if self.record_prompts_dist:
                    self.rollout_prompts[self.locals['infos'][i]['target']][self.locals['infos'][i]['prompt']] += 1

        if self.evaluate:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:                
                t0 = time.time()
                eval_full_score, eval_partial_score, stats = evaluate_model(self.model, self.training_env,
                                            deterministic=self.deterministic,
                                            use_action_masks=self.use_action_masks,
                                            max_number_of_steps=self.max_number_of_steps,
                                            verbose=1,
                                            **self.eval_config)
                print(f'Evaluation took {time.time() - t0:2f} seconds')

                eval_score = max(eval_full_score, eval_partial_score)

                self.logger.record('eval/eval_score', eval_score)
                number_of_targets = len(np.unique(self.eval_config['labels'][:, 0]))
                for i in range(number_of_targets):
                    for key, value in stats[f'target_{i}'].items():
                        if 'percentage' in key:
                            self.logger.record(f'stats/target_{i}/{key}', value)
                
                if self.logs_path is not None:
                    with open(os.path.join(self.logs_path, 'evaluations.json'), 'r+') as log_file:
                        old_data = json.load(log_file)
                        old_data['evaluations'].append({'step': self.num_timesteps,
                                                        'eval_score': eval_score,
                                                        'stats': stats})
                        log_file.seek(0)
                        json.dump(old_data, log_file, indent=4)
                        log_file.truncate()

                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    
                    vec_normalizer = self.model.get_vec_normalize_env()
                    if vec_normalizer:
                        vec_normalizer.save(os.path.join(self.best_model_save_path, "best_model_vec_normalizer.pkl"))
                    self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))

                    print(f'New best model saved with score {self.best_eval_score:2f}')
                else:
                    print(f'Eval score: {eval_score:2f}. Last best score: {self.best_eval_score:2f}')

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        total_steps = self.locals['n_steps'] * self.training_env.num_envs
        if self.record_bad_behaviour:
            self.logger.record('behaviour/out_of_bounds', self.out_of_bounds / total_steps)
            self.logger.record('behaviour/wall_collisions', self.wall_collisions / total_steps)
        if self.record_coins_behaviour:
            self.logger.record('behaviour/good_coins', self.good_coins_behaviour / total_steps)
            self.logger.record('behaviour/bad_coins', self.bad_coins_behaviour / total_steps)

        avg_steps_to_end = sum([sum(self.steps_to_end[i]) / len(self.steps_to_end[i]) if len(self.steps_to_end[i]) > 0 else 0
                                 for i in range(self.training_env.num_envs) ]) / self.training_env.num_envs

        self.logger.record('behaviour/move_wins', self.move_wins / self.endings)
        self.logger.record('behaviour/coin_wins', self.coin_wins / self.endings)
        self.logger.record('behaviour/move_loses', self.move_loses / self.endings)
        self.logger.record('behaviour/coin_loses', self.coin_loses / self.endings)
        self.logger.record('behaviour/avg_steps_to_end', avg_steps_to_end / (self.eval_config['size'] ** 2))
        self.logger.record('behaviour/endings', self.endings / total_steps)

        if self.record_targets_dist:
            rollout_targets_total = sum(self.rollout_targets.values())
            rollout_targets_distribution = {}
            for key, val in self.rollout_targets.items():
                d = round(val / rollout_targets_total, 3)
                rollout_targets_distribution[key] = d
                self.targets[key] += val
                self.logger.record(f'rollout_target_dist/target_{key}', d)
            print(f'Rollout targets distribution: {sorted(rollout_targets_distribution.items())}')

        if self.record_prompts_dist:
            print('Rollout prompts distribution per target:')
            for key, values in self.rollout_prompts.items():
                print(f'{key}: ', end='')
                total = sum(values.values())
                dist = {}
                for k, v in values.items():
                    dist[k] = round(v / total, 3)
                    self.prompts[key][k] += v
                print(dist)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if self.record_targets_dist:
            total = sum(self.targets.values())
            dist = {key: round(value / total, 3) for key, value in sorted(self.targets.items())}
            print(f'Final targets distribution: {dist}')

            print(f'\nFinal targets distribution per env: ')
            for env, targets in self.targets_per_envs.items():
                total = sum(targets.values())
                dist = {key: round(value / total, 3) for key, value in sorted(targets.items())}
                print(f'\t{env}: {dist}')

        if self.record_targets_reset_dist:
            print(f'\nFinal targets restart distribution per env: ')
            for env, targets in self.reset_targets.items():
                total = sum(targets.values())
                dist = {key: round(value / total, 3) for key, value in sorted(targets.items())}
                print(f'\t{env}: {dist}')

        if self.record_prompts_dist:
            print('\nFinal prompts distribution per target:')
            for key, values in self.prompts.items():
                print(f'{key}: ', end='')
                total = sum(values.values())
                dist = {k: round(v / total, 3) for k, v in sorted(values.items())}
                print(dist)
