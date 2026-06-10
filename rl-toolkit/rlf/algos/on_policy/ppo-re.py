from rlf.policies.actor_critic.base_actor_critic import ActorCritic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import copy
from collections import defaultdict
from rlf.algos.on_policy.on_policy_base import OnPolicy
from torch import autograd
import numpy as np
import copy
from learn2learn import clone_module
from model import Actor, Critic, GILD_Network,weights_init_
from collections import defaultdict
from utils import Hot_Plug
from utils import update_model
import contextlib
from rlf.rl import utils
import higher
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class PPO_GILD(OnPolicy):
#     def update(self, rollouts, args=None, beginning=False, t=1):
#         self._compute_returns(rollouts)
#         advantages = rollouts.compute_advantages()

#         use_clipped_value_loss = True

#         log_vals = defaultdict(lambda: 0)

#         for e in range(self._arg('num_epochs')):
#             data_generator = rollouts.get_generator(advantages,
#                     self._arg('num_mini_batch'))

#             for sample in data_generator:
#                 # Get all the data from our batch sample
#                 ac_eval = self.policy.evaluate_actions(sample['state'],
#                         sample['other_state'],
#                         sample['hxs'], sample['mask'],
#                         sample['action'])

#                 ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
#                 surr1 = ratio * sample['adv']
#                 surr2 = torch.clamp(ratio,
#                         1.0 - self._arg('clip_param'),
#                         1.0 + self._arg('clip_param')) * sample['adv']
#                 actor_loss = -torch.min(surr1, surr2).mean(0)

#                 if use_clipped_value_loss:
#                     value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
#                                     -self._arg('clip_param'),
#                                     self._arg('clip_param'))
#                     value_losses = (ac_eval['value'] - sample['return']).pow(2)
#                     value_losses_clipped = (
#                         value_pred_clipped - sample['return']).pow(2)
#                     value_loss = 0.5 * torch.max(value_losses,
#                                                  value_losses_clipped).mean()
#                 else:
#                     value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

#                 loss = (value_loss * self._arg('value_loss_coef') + actor_loss -
#                      ac_eval['ent'].mean() * self._arg('entropy_coef'))
                
#                 # TODO: Add action loss

#                 self._standard_step(loss)

#                 log_vals['value_loss'] += value_loss.sum().item()
#                 log_vals['actor_loss'] += actor_loss.sum().item()
#                 log_vals['dist_entropy'] += ac_eval['ent'].mean().item()
#                 log_vals["policy_update_data"] += self._arg('num_mini_batch')

#         num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
#         for k in log_vals:
#             log_vals[k] /= num_updates

#         log_vals["policy_update_data"] *= num_updates
#         return log_vals

#     def get_add_args(self, parser):
#         super().get_add_args(parser)
#         parser.add_argument(f"--{self.arg_prefix}clip-param",
#             type=float,
#             default=0.2,
#             help='ppo clip parameter')

#         parser.add_argument(f"--{self.arg_prefix}entropy-coef",
#             type=float,
#             default=0.01,
#             help='entropy term coefficient (old default: 0.01)')

#         parser.add_argument(f"--{self.arg_prefix}value-loss-coef",
#             type=float,
#             default=0.5,
#             help='value loss coefficient')
        
#         parser.add_argument(f"--{self.arg_prefix}bc-coef",
#             type=float,
#             default=0.1,
#             help='behavior cloning loss coefficient')

#         parser.add_argument(f"--{self.arg_prefix}gild-coef",
#             type=float,
#             default=0.1,
#             help='gild loss coefficient')

class StepInfo:
    def __init__(self, cur_num_steps: int, cur_num_episodes: int, is_eval: bool):
        self.cur_num_steps = cur_num_steps
        self.cur_num_episodes = cur_num_episodes
        self.is_eval = is_eval

class PPO_GILD(OnPolicy):
    def __init__(self):
        super().__init__()
        self.demo_states = None
        self.demo_actions = None
        self.gild_network = None
        self.gild_optimizer = None
        self.update_iter = 0

    def init(self, policy, args):
        super().init(policy, args)

        # 初始化 GILD 网络（输入维度 = 动作 + 状态 + 动作）
        act_dim = policy.action_space.shape[0]
        ob_dim = sum(space.shape[0] for space in policy.obs_space.spaces.values())
        # ob_dim = policy.obs_space.shape[0]
        gild_input_dim = 22
        self.gild_network = GILD_Network(gild_input_dim).to(args.device)
        self.gild_optimizer = torch.optim.Adam(self.gild_network.parameters(), lr=1e-3,weight_decay= 1e-4)
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

        # 加载示教数据
        self.load_demo_data(args)

    def load_demo_data(self, args):
        # 你可以自定义从文件或函数加载示教数据
        demo_data = torch.load("/home/lwt/DRAIL-ys/expert_datasets/push_partial2.pt")  # 示例：dict(state=..., action=...)
        
        self.demo_actions = demo_data['actions'].to(torch.float32).to(args.device)
        self.demo_states = demo_data['obs'].to(torch.float32).to(args.device)
        self.demo_next_states = demo_data['next_obs'].to(torch.float32).to(args.device)
        self.demo_done = demo_data['done'].to(torch.float32).to(args.device)
        self.demo_ep_found_goal = demo_data['ep_found_goal'].to(torch.float32).to(args.device)

    def get_demo_data(self, batch_size):
        idx = np.random.randint(0, len(self.demo_states), size=batch_size)
        return self.demo_states[idx], self.demo_actions[idx]

    @staticmethod
    def compute_policy_loss(policy, rollout, args, modulation=None):

        ac_eval = policy.evaluate_actions(
            rollout['state'],
            rollout['other_state'],
            rollout['hxs'],
            rollout['mask'],
            rollout['action'],
            modulation=modulation
        )

        ratio = torch.exp(ac_eval['log_prob'] - rollout['prev_log_prob'])

        surr1 = ratio * rollout['adv']
        surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * rollout['adv']
        actor_loss = -torch.min(surr1, surr2).mean(0)

        value_pred_clipped = rollout['value'] + (ac_eval['value'] - rollout['value']).clamp(
            -args.clip_param, args.clip_param)
        value_losses = (ac_eval['value'] - rollout['return']).pow(2)
        value_losses_clipped = (value_pred_clipped - rollout['return']).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        entropy_loss = ac_eval['ent'].mean()

        loss = value_loss * args.value_loss_coef + actor_loss - args.entropy_coef * entropy_loss
        return loss





    def update(self, meta_rollout_bc, meta_rollout_gild, gild_coef, rollouts, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()
        log_vals = defaultdict(lambda: 0)       
        for e in range(self._arg('num_epochs')):                        
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                state_batch = sample['state']
                other_state_batch = sample['other_state']
                hxs_batch = sample['hxs']
                mask_batch = sample['mask']
                action_batch = sample['action']
                prev_log_prob_batch = sample['prev_log_prob']
                return_batch = sample['return']
                value_batch = sample['value']
                adv_batch = sample['adv']

                # ===============================================
                # Step 1: Clone policy
                # ===============================================
                actor_tmp = copy.deepcopy(self.policy)  # 注意：这里需要import copy

                # ===============================================
                # Step 2: Update actor_tmp with RL + BC
                # ===============================================

                # Evaluate actions for actor_tmp
                ac_eval_tmp = actor_tmp.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )

                ratio_tmp = torch.exp(ac_eval_tmp['log_prob'] - prev_log_prob_batch)
                surr1_tmp = ratio_tmp * adv_batch
                surr2_tmp = torch.clamp(ratio_tmp,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * adv_batch
                actor_loss_tmp = -torch.min(surr1_tmp, surr2_tmp).mean()

                # Value loss for actor_tmp
                value_pred_clipped_tmp = value_batch + (ac_eval_tmp['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param'))
                value_losses_tmp = (ac_eval_tmp['value'] - return_batch).pow(2)
                value_losses_clipped_tmp = (value_pred_clipped_tmp - return_batch).pow(2)
                value_loss_tmp = 0.5 * torch.max(value_losses_tmp, value_losses_clipped_tmp).mean()

                # BC loss for actor_tmp
                demo_state, demo_action = self.get_demo_data(len(state_batch))
                bc_loss_tmp = F.mse_loss(actor_tmp.act(demo_state), demo_action)

                total_tmp_loss = (
                    actor_loss_tmp +
                    self._arg('value_loss_coef') * value_loss_tmp +
                    self._arg('bc_coef') * bc_loss_tmp -
                    ac_eval_tmp['ent'].mean() * self._arg('entropy_coef')
                )
                # Optimize actor_tmp manually (without optimizer)
                # 仅获取 actor 的参数
                actor_params = list(actor_tmp.actor.parameters()) + list(actor_tmp.dist.parameters())

                # 计算梯度
                tmp_grads = torch.autograd.grad(
                    total_tmp_loss, actor_params, create_graph=True
                )

                # 更新 actor 的参数
                with torch.no_grad():
                    for p, g in zip(actor_params, tmp_grads):
                        if g is not None:  # 确保梯度存在
                            p.add_(-self._arg('lr') * g)    # 注意：需要确保 self._arg('lr') 是学习率


                # ===============================================
                # Step 3: Update true actor with RL + GILD
                # ===============================================

                ac_eval = self.policy.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )

                ratio = torch.exp(ac_eval['log_prob'] - prev_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss for true actor
                value_pred_clipped = value_batch + (ac_eval['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param'))
                value_losses = (ac_eval['value'] - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # GILD loss
                with torch.no_grad():
                    detached_action = self.policy.act(demo_state).detach()

                gild_input = torch.cat([detached_action, demo_state, demo_action], dim=1)
                gild_output = self.gild_network(gild_input)
                gild_loss = gild_output.mean()

                total_actor_loss = (
                    actor_loss +
                    self._arg('value_loss_coef') * value_loss +
                    gild_coef * gild_loss.detach() -
                    ac_eval['ent'].mean() * self._arg('entropy_coef')
                )

                self._standard_step(total_actor_loss)  # 优化真实 actor
                

                # ===============================================
                # Step 4: Meta-Learning: update GILD Network
                # ===============================================

                # 验证集上测试 actor_tmp 和 actor 的效果
                # 这里为了简单，直接用当前 batch
                # state_val = state_batch
  
                # # compute meta loss
                # # RL loss of actor_tmp (RL+BC)
                # masks = torch.ones(state_val.size(0), 1, device=state_val.device)  # 默认全 1 的 mask
                # action_val_bc = actor_tmp.act(state_val)
                # action_val_bc.requires_grad_(True)  # 确保这个张量需要计算梯度
                # policy_bc_loss_val = -self.policy.critic(state_val, action_val_bc,masks)[1].mean().detach() # -Q_bc
                # # RL loss of actor(RL+GILD)
                # action_val_gild = self.policy.act(state_val)
                # action_val_gild.requires_grad_(True)  # 确保这个张量需要计算梯度
                # policy_loss_val_gild = -self.policy.critic(state_val, action_val_gild,masks)[1].mean() # -Q_gild
                # # meta loss
                # utility = torch.tanh(policy_bc_loss_val - policy_loss_val_gild)  # tanh(Q_gild-Q_bc)
                # meta_loss = -utility # maximize utility


                # # 更新 GILD 网络
                # self.gild_optimizer.zero_grad()
                # meta_loss.backward()
                # self.gild_optimizer.step()

                # # ===============================================
                # # 记录log
                # # ===============================================
                # log_vals['actor_loss'] += actor_loss.item()
                # log_vals['bc_loss'] += bc_loss_tmp.item()
                # log_vals['gild_loss'] += gild_loss.item()
                # log_vals['meta_loss'] += meta_loss.item()
        
        #   ========================New step 4=========================
        train_ctx = torch.no_grad
        num_steps_meta = 127
        bc_episode_count = 0
        gild_episode_count = 0
        #使用bc的策略重新交互收集数据
        for step in range(num_steps_meta):
            bc_obs = meta_rollout_bc.get_obs(step)                   
            bc_step_info = StepInfo(cur_num_steps=(self.update_iter * 128 + step) * 32, cur_num_episodes=bc_episode_count, is_eval=False)
            with train_ctx():
                bc_ac_info = actor_tmp.get_action(
                    utils.get_def_obs(bc_obs, self.args.policy_ob_key),
                    utils.get_other_obs(bc_obs),
                    meta_rollout_bc.get_hidden_state(step),
                    meta_rollout_bc.get_masks(step),
                    bc_step_info,
                )
                if self.args.clip_actions:
                    bc_ac_info.clip_action(*self.ac_tensor)
            bc_next_obs, bc_reward, bc_done, bc_infos = self.envs.step(bc_ac_info.take_action)
            bc_reward += bc_ac_info.add_reward          
            bc_episode_count += sum([int(d) for d in bc_done])
            bc_done = torch.tensor(bc_done.reshape(-1, 1), dtype=torch.bool)
            meta_rollout_bc.insert(bc_obs, bc_next_obs, bc_reward, bc_done, bc_infos, bc_ac_info)    

        #使用gild的策略重新交互收集数据
        for step in range(num_steps_meta):
            gild_obs = meta_rollout_gild.get_obs(step)                   
            gild_step_info = StepInfo(cur_num_steps=(self.update_iter * 128 + step) * 32, cur_num_episodes=gild_episode_count, is_eval=False)
            with train_ctx():
                gild_ac_info = self.policy.get_action(
                    utils.get_def_obs(gild_obs, self.args.policy_ob_key),
                    utils.get_other_obs(gild_obs),
                    meta_rollout_gild.get_hidden_state(step),
                    meta_rollout_gild.get_masks(step),
                    gild_step_info,
                )
                if self.args.clip_actions:
                    gild_ac_info.clip_action(*self.ac_tensor)
            gild_next_obs, gild_reward, gild_done, gild_infos = self.envs.step(gild_ac_info.take_action)
            gild_reward += gild_ac_info.add_reward          
            gild_episode_count += sum([int(d) for d in gild_done])
            gild_done = torch.tensor(gild_done.reshape(-1, 1), dtype=torch.bool)
            meta_rollout_gild.insert(gild_obs, gild_next_obs, gild_reward, gild_done, gild_infos, gild_ac_info)   
        
        """
        使用来自 RL+BC 和 RL+GILD 的 rollout 数据，执行 GILD 网络的元学习更新。
        """
        self._compute_returns(meta_rollout_bc)
        self._compute_returns(meta_rollout_gild)

        advantages_bc = meta_rollout_bc.compute_advantages()
        advantages_gild = meta_rollout_gild.compute_advantages()

        bc_gen = meta_rollout_bc.get_generator(advantages_bc, self._arg('num_mini_batch'))
        gild_gen = meta_rollout_gild.get_generator(advantages_gild, self._arg('num_mini_batch'))

        # 使用对应 mini-batch 执行一次 meta-update
        for sample_bc, sample_gild in zip(bc_gen, gild_gen):
            # 1. 使用 higher 对 policy 做 differentiable inner loop
            with higher.innerloop_ctx(self.policy, self.actor_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                
                detached_action = self.policy.act(sample_gild['state'])

                gild_input = torch.cat([
                    detached_action,
                    sample_gild['state'],
                    sample_gild['action']  # 或 sample_gild['demo_action']，具体按你任务数据结构
                ], dim=1)
                # gild_input = sample_gild['state']  # 假设用于 gild_network 的输入
                mod_output = self.gild_network(gild_input)

                # 构造一个调制过的 loss（作为 GILD 调控信号）
                loss_gild_inner = self.compute_policy_loss(fnet, sample_gild, self.args, modulation=mod_output) 

                diffopt.step(loss_gild_inner)
            
            policy_loss_val_gild = self.compute_policy_loss(fnet, sample_gild, self.args, modulation=mod_output)

            # 2. 计算 bc 分支的 loss（无需 higher）
            policy_loss_val_bc = self.compute_policy_loss(actor_tmp, sample_bc, self.args)

            # 3. 元损失定义为二者差值（越大越好）
            delta = policy_loss_val_bc - policy_loss_val_gild
            utility = torch.tanh(delta)
            loss_meta = -utility

            self.gild_optimizer.zero_grad()
            loss_meta.backward()
            self.gild_optimizer.step()
            
        # ===============================================
                

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        return log_vals



    def get_add_args(self, parser):
        super().get_add_args(parser)
        prefix = self.arg_prefix
        parser.add_argument(f"--{prefix}bc-coef", type=float, default=0.1,
                        help='behavior cloning loss coefficient')
        parser.add_argument(f"--{prefix}gild-coef", type=float, default=0.1,
                        help='gild loss coefficient')
        parser.add_argument(f"--{prefix}demo-data-path", type=str, required=True,
                        help='Path to the demonstration data (torch file)')
        parser.add_argument(f"--{self.arg_prefix}clip-param",
            type=float,
            default=0.2,
            help='ppo clip parameter')

        parser.add_argument(f"--{self.arg_prefix}entropy-coef",
            type=float,
            default=0.01,
            help='entropy term coefficient (old default: 0.01)')
        parser.add_argument(
            f"--{self.arg_prefix}value-loss-coef",
            type=float,
            default=0.5,
            help="Coefficient for value loss",
        )
        parser.add_argument(
            f"--{self.arg_prefix}lr",
            type=float,
            default=3e-4,
            help="Learning rate for actor in PPO_GILD"
        )

