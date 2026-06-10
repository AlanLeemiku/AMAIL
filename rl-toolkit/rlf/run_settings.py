import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from gym.spaces import Box

import rlf
import rlf.rl.utils as rutils
from rlf.args import get_default_parser
from rlf.envs.env_interface import get_env_interface
from rlf.exp_mgr import config_mgr
from rlf.il.traj_mgr import TrajSaver
from rlf.rl.checkpointer import Checkpointer
from rlf.rl.envs import make_vec_envs
from rlf.rl.evaluation import full_eval
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.runner import Runner

from collections import deque

class replaybuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, next_obs, reward, done):
        self.buffer.append((obs, action, next_obs, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, next_obs, reward, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )
    def insert(self, obs, next_obs, rewards, done, info, ac_info):
        super().insert(obs, next_obs, rewards, done, info, ac_info)
        masks, bad_masks = self.compute_masks(done, info)

        for k in self.ob_keys:
            if k is None:
                self.obs[self.step + 1].copy_(next_obs)
            else:
                self.obs[k][self.step + 1].copy_(next_obs[k])

        for i, inf in enumerate(info):
            for k in self.get_extract_info_keys():
                if k in inf:
                    if not isinstance(inf[k], torch.Tensor):
                        assign_val = torch.tensor(inf[k]).to(self.args.device)
                    else:
                        assign_val = inf[k]
                    self.add_data[k][self.step, i] = assign_val

        self.actions[self.step].copy_(ac_info.action)
        self.action_log_probs[self.step].copy_(ac_info.action_log_probs)
        self.value_preds[self.step].copy_(ac_info.value)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        for k in self.hidden_states:
            self.hidden_states[k][self.step + 1].copy_(ac_info.hxs[k])

        self.step = (self.step + 1) % self.num_steps    
    def __len__(self):
        return len(self.buffer)

    import random
from collections import deque
import numpy as np
import torch

class replaybuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        """
        初始化经验回放池
        参数：
            capacity: 最大缓存容量（条数）
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)  # 使用 deque 实现固定长度的 FIFO 缓冲区

    def push(self, obs, action, next_obs, reward, done):
        """
        添加一条经验到缓冲区中
        参数：
            obs: 当前状态
            action: 执行动作
            next_obs: 执行动作后的下一个状态
            reward: 奖励值
            done: 是否结束标志
        """
        self.buffer.append((obs, action, next_obs, reward, done))  # 将五元组加入缓冲区

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个 batch
        参数：
            batch_size: 采样的样本数
        返回：
            五个张量:状态、动作、下一个状态、奖励、done 标志
        """
        batch = random.sample(self.buffer, batch_size)  # 随机采样
        obs, action, next_obs, reward, done = map(np.stack, zip(*batch))  # 转置并堆叠成矩阵

        # 转换为 PyTorch 张量并返回
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),  # 奖励和 done 加一个维度变成列向量
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
        )

    # def insert(self, obs, next_obs, rewards, done, info, ac_info):
    #     """
    #     ⚠️ 此函数为 on-policy 策略（如 PPO）中的插入逻辑，通常不属于经验回放池。
    #     如果你是在实现 DDPG/SAC 等 off-policy 算法，请删除或移动此函数。
    #     """
    #     # 错误：当前类未继承任何父类，调用 super() 会报错
    #     super().insert(obs, next_obs, rewards, done, info, ac_info)

    #     # 计算 masks（处理 episode 是否终止）
    #     masks, bad_masks = self.compute_masks(done, info)

    #     # 存储观察值（支持带键的观测结构）
    #     for k in self.ob_keys:
    #         if k is None:
    #             self.obs[self.step + 1].copy_(next_obs)
    #         else:
    #             self.obs[k][self.step + 1].copy_(next_obs[k])

    #     # 存储 info 字典中额外信息
    #     for i, inf in enumerate(info):
    #         for k in self.get_extract_info_keys():
    #             if k in inf:
    #                 if not isinstance(inf[k], torch.Tensor):
    #                     assign_val = torch.tensor(inf[k]).to(self.args.device)
    #                 else:
    #                     assign_val = inf[k]
    #                 self.add_data[k][self.step, i] = assign_val

    #     # 存储动作、log_prob、价值、奖励、mask 等 rollout 数据
    #     self.actions[self.step].copy_(ac_info.action)
    #     self.action_log_probs[self.step].copy_(ac_info.action_log_probs)
    #     self.value_preds[self.step].copy_(ac_info.value)
    #     self.rewards[self.step].copy_(rewards)
    #     self.masks[self.step + 1].copy_(masks)
    #     self.bad_masks[self.step + 1].copy_(bad_masks)

    #     # 存储 RNN 隐状态
    #     for k in self.hidden_states:
    #         self.hidden_states[k][self.step + 1].copy_(ac_info.hxs[k])

    #     # 步数递增（rollout buffer 特有）
    #     self.step = (self.step + 1) % self.num_steps

    def __len__(self):
        """
        返回当前缓存中经验数量
        """
        return len(self.buffer)


def init_seeds(args):
    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.set_num_threads(1)


try:
    from ray import tune

    MasterClass = tune.Trainable
except:

    class BlankTrainable:
        def __init__(self, config, logger_creator):
            pass

    MasterClass = BlankTrainable

class RunSettings(MasterClass):
    """
    Sets up the training, environments, and all other information needed for
    running an algorithm.
    """

    def __init__(self, args_str: str = None, config=None, logger_creator=None):
        """
        Args:
        :param args_str: Parse the arguments from this string.
        :param config: The config for tune.Trainable if used.
        :param logger_creator: Also used for tune.Trainble if used.
        """
        self._preset_args = None if args_str is None else args_str.split(" ")
        self.working_dir = os.getcwd()

        base_parser = self._get_base_parser()
        if self._preset_args is None:
            self.base_args, _ = base_parser.parse_known_args()
        else:
            self.base_args, _ = base_parser.parse_known_args(self._preset_args)
        super().__init__(config, logger_creator)

    def _get_base_parser(self):
        base_parser = argparse.ArgumentParser()
        self.get_add_args(base_parser)
        return base_parser

    def get_config_file(self) -> str:
        """
        :return: The location to a config file that holds whatever information
            about the project.
        """
        return osp.join(self.working_dir, "config.yaml")

    def create_traj_saver(self, save_path: str) -> rlf.il.TrajSaver:
        """
        How trajectories should be saved if desired.
        :save_path: file name to write the trajectories to
        """
        return TrajSaver(save_path)

    def get_add_args(self, parser):
        pass

    def get_logger(self):
        return BaseLogger()

    def get_add_ray_config(self, config):
        return config

    def get_add_ray_kwargs(self):
        return {}

    def get_policy(self) -> rlf.policies.BasePolicy:
        """
        :return: The policy for training
        """
        raise NotImplementedError("Must return policy to be used.")

    def get_algo(self) -> rlf.algos.BaseAlgo:
        """
        :return: The algorithm to update the policy with.
        """
        raise NotImplementedError("Must return algorithm to be used")

    def _get_env_interface(self, args, task_id=None):
        env_interface = get_env_interface(args.env_name)(args)
        env_interface.setup(args, task_id)
        return env_interface

    def get_parser(self):
        return get_default_parser()

    def get_args(self, algo, policy):
        parser = self.get_parser()
        algo.get_add_args(parser)
        policy.get_add_args(parser)

        if self._preset_args is None:
            args, rest = parser.parse_known_args()
        else:
            args, rest = parser.parse_known_args(self._preset_args)

        env_parser = argparse.ArgumentParser()
        get_env_interface(args.env_name)(args).get_add_args(env_parser)
        env_args, rest = env_parser.parse_known_args(rest)
        # Assign the env args to the main args namespace.
        rutils.update_args(args, vars(env_args))

        # Check that there are no arguments not accounted for in `base_args`
        _, rest_of_args = self._get_base_parser().parse_known_args(rest)
        if "-v" in rest_of_args:
            del rest_of_args[rest_of_args.index("-v")]
            print("Env args:")
            env_parser.print_help()
            print("Alg args:")
            parser.print_help()
            sys.exit(0)
        if len(rest_of_args) != 0:
            raise ValueError("Unrecognized arguments %s" % str(rest_of_args))

        # Convert the types of some of the standard types that don't allow the
        # scientific notation when expecting integer inputs.
        args.num_env_steps = int(args.num_env_steps)
        return args

    def stop(self):
        self.ray_runner.close()
        del self.ray_runner
        del self.ray_args

    def _sys_setup(self, add_args, ray_create, algo, policy):
        # Set up args used for training
        args = self.get_args(algo, policy)
        args.cwd = self.working_dir
        if "wandb" in add_args:
            del add_args["wandb"]
        rutils.update_args(args, add_args, True)
        if "cwd" in add_args:
            self.working_dir = add_args["cwd"]

        config_mgr.init(self.get_config_file())
        if args.ray:
            # No logger when ray is tuning
            log = BaseLogger()
        else:
            if ray_create:
                return None, None
            log = self.get_logger()
        for k, v in vars(self.base_args).items():
            if k not in args:
                setattr(args, k, v)
        log.init(args)
        log.set_prefix(args)

        args.device = torch.device("cuda:0" if args.cuda else "cpu")
        init_seeds(args)
        if args.detect_nan:
            torch.autograd.set_detect_anomaly(True)
        return args, log

    def create_runner(self, add_args={}, ray_create=False) -> rlf.Runner:
        """
        Gets the runner used for training.
        """
        policy = self.get_policy()
        algo = self.get_algo()

        args, log = self._sys_setup(add_args, ray_create, algo, policy)
        if args is None:
            return None
        env_interface = self._get_env_interface(args)

        checkpointer = Checkpointer(args)

        alg_env_settings = algo.get_env_settings(args)
        print("args.eval_only:", args.eval_only)
        #import ipdb; ipdb.set_trace()
        # Setup environment
        _, envs = make_vec_envs(
            args.env_name,
            args.seed,
            args.num_processes,
            args.gamma,
            args.device,
            True,
            env_interface,
            args,
            alg_env_settings,
            #set_eval=args.eval_only,
            False
        )

        # 获取环境的观察空间和动作空间维度
        obs_dim = envs.observation_space.shape[0] if isinstance(envs.observation_space, Box) else None
        action_dim = envs.action_space.shape[0] if isinstance(envs.action_space, Box) else None

        # 创建 ReplayBuffer
        replay_buffer = replaybuffer(capacity=100000, obs_dim=obs_dim, action_dim=action_dim)
        
        rutils.pstart_sep()
        print("Action space:", envs.action_space)
        if isinstance(envs.action_space, Box):
            print("Action range:", (envs.action_space.low, envs.action_space.high))
        print("Observation space", envs.observation_space)
        rutils.pend_sep()

        # Setup policy
        policy_args = (envs.observation_space, envs.action_space, args)
        policy.init(*policy_args)
        policy = policy.to(args.device)
        policy.watch(log)
        policy.set_env_ref(envs)

        # Setup algo
        algo.set_get_policy(self.get_policy, policy_args)
        algo.set_env_ref(envs)
        #import ipdb; ipdb.set_trace()
        algo.init(policy, args)

        # Setup storage buffer
        storage = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            storage.add_info_key(ik, get_shape(envs))
        storage.to(args.device)
        storage.init_storage(envs.reset())
        storage.set_traj_done_callback(algo.on_traj_finished)

        #设置GILD更新使用的两个存储器
        meta_rollout_bc = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            meta_rollout_bc.add_info_key(ik, get_shape(envs))
        meta_rollout_bc.to(args.device)
        meta_rollout_bc.init_storage(envs.reset())
        meta_rollout_bc.set_traj_done_callback(algo.on_traj_finished)
        
        meta_rollout_gild = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            meta_rollout_gild.add_info_key(ik, get_shape(envs))
        meta_rollout_gild.to(args.device)
        meta_rollout_gild.init_storage(envs.reset())
        meta_rollout_gild.set_traj_done_callback(algo.on_traj_finished)
        
        
        simple_env = ['Sine-v0', 'SCurve-v0', 'Dalmatian-v0', 'Triangle-v0', 'Triangle-v2', 'Rectangle-v0', 'Rectangle-v1']

        runner = self._get_runner_cls(algo, policy)(
            envs, storage, policy, log, env_interface, checkpointer, args, algo, meta_rollout_bc,meta_rollout_gild
        )
        
        return runner

    def _get_runner_cls(self, algo, policy):
        return Runner

    def import_add(self):
        """
        Needed for ray training.
        """
        pass

    def setup(self, config):
        """
        Only called during ray training.
        """
        self.import_add()
        self.ray_runner = self.create_runner(config, ray_create=True)
        if self.ray_runner is None:
            return
        self.ray_runner.setup()
        self.ray_args = self.ray_runner.args

        if not self.ray_args.ray_debug:
            self.ray_runner.log.disable_print()

    def step(self):
        """
        Only called during ray training
        """
        updater_log_vals = self.ray_runner.training_iter(self.training_iteration)
        if (self.training_iteration + 1) % self.ray_args.log_interval == 0:
            log_dict = self.ray_runner.log_vals(
                updater_log_vals, self.training_iteration
            )
        if (self.training_iteration + 1) % self.ray_args.save_interval == 0:
            self.ray_runner.save(self.training_iteration)
        if (self.training_iteration + 1) % self.ray_args.eval_interval == 0:
            self.ray_runner.eval(self.training_iteration)

        return log_dict
