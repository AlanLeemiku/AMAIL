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
import learn2learn as l2l
from learn2learn import clone_module
from model import Actor, Critic, GILD_Network,weights_init_
from collections import defaultdict
from utils import Hot_Plug
from utils import update_model
import contextlib
from fuzzy_generator import FuzzyGenerator
from rlf.rl import utils
import gc
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self._scale = 1.0
        self._scale_initialized = False 

    def init(self, policy, args):
        super().init(policy, args)

        # 初始化 GILD 网络（输入维度 = 动作 + 状态 + 动作）
        # act_dim = policy.action_space.shape[0]
        # ob_dim = sum(space.shape[0] for space in policy.obs_space.spaces.values())
        # ob_dim = policy.obs_space.shape[0]
        gild_input_dim = 148  #ant
        # gild_input_dim = 24  #pick
        # gild_input_dim = 22  #push
        self.gild_network = GILD_Network(gild_input_dim).to(args.device)
        self.gild_optimizer = torch.optim.Adam(self.gild_network.parameters(), lr=1e-3,weight_decay= 1e-4)
 
        # 加载示教数据
        self.load_demo_data(args)
        # argparse converts option names with '-' to attributes with '_' (e.g. --drail-fuzzy-path -> args.drail_fuzzy_path)
        fuzzy_path = getattr(args, f"{self.arg_prefix}_fuzzy_path", None)
        fuzzy_R = getattr(args, f"{self.arg_prefix}_fuzzy_R", 256)

    def load_demo_data(self, args):
        """加载示教数据，处理可能缺失的键"""
        # 从配置中获取数据路径，默认使用硬编码路径
        demo_path = getattr(args, 'traj_load_path', "/home/lwt/DRAIL-ys/expert_datasets/push_partial2.pt")
        demo_data = torch.load(demo_path)
        
        # 加载必需的数据
        self.demo_actions = demo_data['actions'].to(torch.float32).to(args.device)
        self.demo_states = demo_data['obs'].to(torch.float32).to(args.device)
        
        # 加载可选的数据，如果不存在则使用默认值
        if 'next_obs' in demo_data:
            self.demo_next_states = demo_data['next_obs'].to(torch.float32).to(args.device)
        else:
            self.demo_next_states = None
            
        if 'done' in demo_data:
            self.demo_done = demo_data['done'].to(torch.float32).to(args.device)
        else:
            self.demo_done = None
            
        # ep_found_goal 可能不存在（例如在 maze 环境中），提供默认值
        if 'ep_found_goal' in demo_data:
            self.demo_ep_found_goal = demo_data['ep_found_goal'].to(torch.float32).to(args.device)
        else:
            # 如果不存在，创建一个全零张量或全一张量作为默认值
            self.demo_ep_found_goal = torch.zeros(len(self.demo_states), dtype=torch.float32).to(args.device)

    def get_demo_data(self, batch_size):
        idx = np.random.randint(0, len(self.demo_states), size=batch_size)
        return self.demo_states[idx], self.demo_actions[idx]



    def update(self, meta_rollout_bc, meta_rollout_gild, gild_coef, discrim ,rollouts, args=None, beginning=False, t=1):
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
                
                actor_tmp = l2l.clone_module(self.policy)
                # 确保 actor_tmp 的参数是叶子节点
                for param in actor_tmp.parameters():
                    param.detach_().requires_grad_()
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
                demo_action = demo_action.detach() # 确保 demo_action 可微分（去掉不必要的 requires_grad_(True)）
                demo_state = demo_state.detach() # 确保 demo_state 可微分
                demo_action_shape = demo_action.shape
                demo_state_shape = demo_state.shape
                bc_loss_tmp = F.mse_loss(actor_tmp.act(demo_state), demo_action)
                total_tmp_loss = (
                    actor_loss_tmp +
                    self._arg('value_loss_coef') * value_loss_tmp +
                    self._arg('bc_coef') * bc_loss_tmp -
                    ac_eval_tmp['ent'].mean() * self._arg('entropy_coef')
                )
                
                policy_bc_loss_val = total_tmp_loss.detach()
                # # Optimize actor_tmp manually (without optimizer)
                # # 仅获取 actor 的参数
                # actor_params = list(actor_tmp.actor.parameters()) + list(actor_tmp.dist.parameters())

                # # 计算梯度
                # tmp_grads = torch.autograd.grad(
                #     total_tmp_loss, actor_params, create_graph=True
                # )

                # # 更新 actor 的参数
                # with torch.no_grad():
                #     for p, g in zip(actor_params, tmp_grads):
                #         if g is not None:  # 确保梯度存在
                #             p.add_(-self._arg('lr') * g)    # 注意：需要确保 self._arg('lr') 是学习率
                # BC loss for actor_tmp
                torch.autograd.set_detect_anomaly(True)  # 启用异常检测
                # 使用优化器更新 actor_tmp
                tmp_optimizer = torch.optim.Adam(actor_tmp.parameters(), lr=1e-1)  # 增大学习率
                # grads = torch.autograd.grad(total_tmp_loss, actor_tmp.parameters(), create_graph=True)
                # with torch.no_grad():
                #     for p, g in zip(actor_tmp.parameters(), grads):
                #         if g is not None:
                #             p.add_(-1e-1 * g)  # 用你设定的学习率
                tmp_optimizer.zero_grad()
                total_tmp_loss.backward()
                tmp_optimizer.step()

                # 检查参数是否更新
                # for param in actor_tmp.parameters():
                #     print(param)  # 检查参数是否变化
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
                
                action_for_gild = self.policy.act(demo_state)

                gild_input = torch.cat([action_for_gild, demo_state, demo_action], dim=1)
                # print(gild_input)  # 检查 gild_input 是否变化
                gild_output = self.gild_network(gild_input)
                # print(gild_output)  # 检查 gild_input 是否变化
                # gild_loss = gild_output.mean()
                
                
                with torch.no_grad():
                    # 1. 构造专家轨迹
                    expert_traj = torch.cat([demo_state, demo_action], dim=1)  # [batch, obs_dim + act_dim]
                    num_inputs = self.demo_states.shape[1]

                    # 2. 策略网络输出均值和标准差
                    dist, _, _ = self.policy.forward(expert_traj[:, :num_inputs], None, None, None)
                    ac_mean = dist.mean
                    ac_std = dist.stddev
                    ac = expert_traj[:, num_inputs:]
                    ac_var = ac_std ** 2
                    lg_prob = -((ac - ac_mean) ** 2) / (2 * ac_var) - torch.log(ac_std) - math.log(math.sqrt(2 * math.pi))
                    prob = torch.exp(lg_prob.sum(-1, keepdim=True))

                    # 3. 判别器输出专家概率
                    disc_val = torch.sigmoid(discrim._compute_disc_val(demo_state, demo_action))  # shape: [batch, 1]
                    disc_val = disc_val.clamp(min=1e-6, max=1-1e-6)  # 防止数值溢出
                    #需要引入 drail_discrim 模块来调用
                    # 4. 置信度公式
                    beta = 1.0
                    expert_conf = ((1 / disc_val - 1) * prob).pow(1 / (beta + 1))* self._scale
                    expert_conf_mean = expert_conf.mean().item()
                    # gild_coef = expert_conf_mean
                    
                    base = (1 / disc_val - 1) * prob.pow(1 / (beta + 1))

                    if not self._scale_initialized:
                        target_mean = 0.5
                        # 计算当前 expert_conf 的均值
                        if expert_conf_mean > 0:
                            # 反推 scale，使 expert_conf 的均值接近 target_mean
                            # scale = target_mean / expert_conf_mean
                            new_scale = target_mean / (expert_conf_mean + 1e-8)
                            self._scale = new_scale
                            print(f"[Auto Scale] expert_conf_mean={expert_conf_mean:.4f}, set scale={self._scale:.4f}")
                            # 用新 scale 再算一次
                            expert_conf = (base * self._scale).pow(1 / (beta + 1))
                            expert_conf_mean = expert_conf.mean().item()
                            print(f"[Auto Scale] after scaling, expert_conf_mean={expert_conf_mean:.4f}")
                        self._scale_initialized = True
                                      
                weighted_gild_loss = (gild_output * expert_conf).mean()
                
                

                total_actor_loss = (
                    actor_loss+
                    self._arg('value_loss_coef') * value_loss +
                    weighted_gild_loss -
                    ac_eval['ent'].mean() * self._arg('entropy_coef')
                )
                #********************将gild网络的更新提前***********************
                
                policy_loss_val_gild= total_actor_loss
                # with torch.no_grad():
                    # for param in self.gild_network.parameters():
                        # param.add_(torch.randn_like(param) * 0.01)

                #**********************将gild网络的更新提前*****************
                self._standard_step(total_actor_loss)  # 优化真实 actor
                
                # ===============================================
                # Step 4: Meta-Learning: update GILD Network
                # ===============================================

                # 验证集上测试 actor_tmp 和 actor 的效果
                # state_val = state_batch
                # masks = torch.ones(state_val.size(0), 1, device=state_val.device)  # 默认全 1 的 mask

                # # 动作生成
                # action_val_bc = actor_tmp.act(state_val)
                # action_val_gild = self.policy.act(state_val)
                # # print("action_val_bc:", action_val_bc)
                # # print("action_val_gild:", action_val_gild)
                
                
                # # 验证集上测试 actor_tmp 和 actor 的效果
                # advantage_bc = self.policy.get_value_for_meta(state_val, action_val_bc)
                # advantage_gild = self.policy.get_value_for_meta(state_val, action_val_gild)
                
                # # 计算 meta loss
                # utility = torch.tanh(advantage_bc[1].mean() - advantage_gild[1].mean())  # 使用优势函数计算效用
                # # 使用 evaluate_actions 计算 RL 损失
                # ac_eval_bc = self.policy.evaluate_actions(state_val, None, None, masks, action_val_bc)
                # policy_bc_loss_val = -ac_eval_bc['value'].mean()  # -Q_bc

                # ac_eval_gild = self.policy.evaluate_actions(state_val, None, None, masks, action_val_gild)
                # policy_loss_val_gild = -ac_eval_gild['value'].mean()  # -Q_gild
                # print(policy_bc_loss_val.item(), policy_loss_val_gild.item())
                # # 计算 meta loss
                # utility = torch.tanh(policy_bc_loss_val - policy_loss_val_gild)  # tanh(Q_gild - Q_bc)
                # meta_loss = -utility  # maximize utility
                # # 更新 GILD 网络
                # self.gild_optimizer.zero_grad()
                # meta_loss.backward()                                  
                # self.gild_optimizer.step()
                
                # with torch.no_grad():
                #     for param in self.gild_network.parameters():
                #         param.add_(torch.randn_like(param) * 0.01)
                
                utility = torch.tanh(policy_bc_loss_val - policy_loss_val_gild)  
                meta_loss = -utility  # maximize utility
                # 更新 GILD 网络
                self.gild_optimizer.zero_grad()
                grad_omega = autograd.grad(meta_loss, self.gild_network.parameters())
                for gradient, variable in zip(grad_omega, self.gild_network.parameters()):
                    variable.grad.data = gradient                               
                self.gild_optimizer.step()
                
                
                #GILD梯度检查
                # for name, param in self.gild_network.named_parameters():
                #     if param.grad is not None:
                #         print(f"GILD grad for {name}: {param.grad.norm().item()}")
                #     else:
                #         print(f"GILD grad for {name}: None")
                
                
                # ===============================================
                # 记录log
                # ===============================================
                log_vals['actor_loss'] += actor_loss.item()
                log_vals['bc_loss'] += bc_loss_tmp.item()
                log_vals['gild_loss'] += weighted_gild_loss.item()
                log_vals['meta_loss'] += meta_loss.item()
                log_vals['bc_loss_val'] += policy_bc_loss_val.item()
                log_vals['gild_loss_val'] += policy_loss_val_gild.item()
                log_vals['gild_coef'] += expert_conf_mean
                # # 清理 actor_tmp 的梯度，防止显存泄漏
                # for p in actor_tmp.parameters():
                #     p.grad = None

                # # 清理 policy 的梯度，防止显存泄漏
                # for p in self.policy.parameters():
                #     p.grad = None

                # # 清理 gild_network 的梯度，防止显存泄漏
                # for p in self.gild_network.parameters():
                #     p.grad = None
                del actor_tmp
                del total_actor_loss
                del policy_bc_loss_val
                del policy_loss_val_gild

        
        print("gild_coef:", expert_conf_mean)
        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        return log_vals



    def get_add_args(self, parser):
        super().get_add_args(parser)
        prefix = self.arg_prefix
        parser.add_argument(f"--{prefix}bc-coef", type=float, default=0.025,
                        help='behavior cloning loss coefficient')
        parser.add_argument(f"--{prefix}gild-coef", type=float, default=0.1,
                        help='gild loss coefficient')
        parser.add_argument(f"--{prefix}demo-data-path", type=str, required=True,
                        help='Path to the demonstration data (torch file)')
        parser.add_argument(f"--{prefix}-fuzzy-path", type=str, default=None,
            help='Path to pretrained fuzzy checkpoint (optional)')
        parser.add_argument(f"--{prefix}-fuzzy-R", type=int, default=256,
            help='Fuzzy R (rules) when loading/initializing fuzzy')
        # Also accept global fuzzy args without prefix for convenience
        parser.add_argument("--fuzzy-path", type=str, default=None,
            help='(global) Path to pretrained fuzzy checkpoint (optional)')
        parser.add_argument("--fuzzy-R", type=int, default=256,
            help='(global) Fuzzy R (rules) when loading/initializing fuzzy')
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

class PPO(OnPolicy):
    def update(self, rollouts, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        use_clipped_value_loss = True

        log_vals = defaultdict(lambda: 0)

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                # Get all the data from our batch sample
                ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['other_state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])

                ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
                surr1 = ratio * sample['adv']
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * sample['adv']
                actor_loss = -torch.min(surr1, surr2).mean(0)

                if use_clipped_value_loss:
                    value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                                    -self._arg('clip_param'),
                                    self._arg('clip_param'))
                    value_losses = (ac_eval['value'] - sample['return']).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

                loss = (value_loss * self._arg('value_loss_coef') + actor_loss -
                     ac_eval['ent'].mean() * self._arg('entropy_coef'))
                
                # TODO: Add action loss

                self._standard_step(loss)

                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['actor_loss'] += actor_loss.sum().item()
                log_vals['dist_entropy'] += ac_eval['ent'].mean().item()
                log_vals["policy_update_data"] += self._arg('num_mini_batch')

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        log_vals["policy_update_data"] *= num_updates
        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}clip-param",
            type=float,
            default=0.2,
            help='ppo clip parameter')

        parser.add_argument(f"--{self.arg_prefix}entropy-coef",
            type=float,
            default=0.01,
            help='entropy term coefficient (old default: 0.01)')

        parser.add_argument(f"--{self.arg_prefix}value-loss-coef",
            type=float,
            default=0.5,
            help='value loss coefficient')