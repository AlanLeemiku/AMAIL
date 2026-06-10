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
from learn2learn.utils import update_module
from model import Actor, Critic, GILD_Network,weights_init_
from collections import defaultdict
from utils import Hot_Plug
from utils import update_model
import contextlib
from rlf.rl import utils
import gc
import math



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
        # buffers for robust one-time scale calibration
        self._gild_scale_values_buf = []
        self._gild_scale_gilds_buf = []
        self._gild_scale_collect_batches = 5
        self._gild_warmup_updates = 2000
        self._gild_warmup_reported = False
        self._gild_min_exit_updates = 600
        self._gild_exit_patience = 20
        self._gild_exit_counter = 0
        self._gild_exit_ratio_threshold = 0.1
        self._gild_exit_utility_threshold = 0.02
        self._gild_exit_disc_threshold = 0.05
        self._gild_anneal_updates = 500
        self._gild_anneal_start_update = None
        self._gild_meta_margin = 0.5
        self._meta_target_gap_ratio = 0.1
        self._meta_signal_coef = 1.0
        self._gild_signal_temp = 0.5
        self._gild_actor_coef = 50.0
        self._base_floor = 1e-8
        self._expert_conf_floor = 1e-8
        self._conf_norm_eps = 1e-4
        self._prob_log_clip_min = -20.0
        self._prob_log_clip_max = 20.0
        self._gild_collapse_eps = 1e-6
        self._gild_collapse_patience = 20
        self._gild_collapse_report_every = 50
        self._gild_collapse_count = 0

    def init(self, policy, args):
        super().init(policy, args)
        
        
        # 初始化 GILD 网络（输入维度 = 动作 + 状态 + 动作）
        gild_input_dim = 24  # pick
        # gild_input_dim = 22  # push
        # gild_input_dim = 148  #ant
        self.gild_network = GILD_Network(gild_input_dim).to(args.device)
        self.gild_optimizer = torch.optim.Adam(self.gild_network.parameters(), lr=2e-3, weight_decay=0.0)
        # Hot_Plug 用于在保留计算图的情况下临时应用 actor 更新以进行 meta 评估
        self.hotplug = Hot_Plug(self.policy)
        self.load_demo_data(args)

    def load_demo_data(self, args):
        demo_data = torch.load(args.demo_data_path)
        self.demo_actions = demo_data['actions'].to(torch.float32).to(args.device)
        self.demo_states = demo_data['obs'].to(torch.float32).to(args.device)
        self.demo_next_states = demo_data['next_obs'].to(torch.float32).to(args.device)
        self.demo_done = demo_data['done'].to(torch.float32).to(args.device)

    def get_demo_data(self, batch_size):
        idx = np.random.randint(0, len(self.demo_states), size=batch_size)
        return self.demo_states[idx], self.demo_actions[idx]

    def update(self, discrim ,rollouts, gild_coef, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        log_vals = defaultdict(lambda: 0)

        # ====== 函数高层说明（update 的总体流程） ======
        # Step 0: 计算 returns 与 advantages（已完成）
        # Step 1: 用 clone(policy) 构造 baseline 内环 (PPO+BC)，对 clone 做一次非可微内环更新并评估 baseline 性能
        # Step 2: 用另一个 clone(policy) 构造 GILD 内环 (PPO+GILD)，对该 clone 做一次可微内环更新（create_graph=True），评估 gild 路径性能
        # Step 3: 对真实策略执行 PPO+GILD 的真实 update（self._standard_step），这是真正影响训练的更新
        # Step 4: 使用 Step1 的 baseline（detach）与 Step2 的 gild 评估结果构造 meta objective，使 meta 梯度通过可微内环回传到 gild_network
        # Step 5: 记录日志与清理临时对象
        # 备注：clone 用于“试验性/模拟性”更新，避免在真实策略上做 apply-then-undo。

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(
                advantages,
                self._arg('num_mini_batch')
            )

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

                # 先抽示教 batch，并基于当前真实策略计算一次置信度权重。
                # 这个权重会同时给 Step 1/2 使用，因此必须放在内环之前。
                demo_state, demo_action = self.get_demo_data(len(state_batch))
                demo_action = demo_action.detach()
                demo_state = demo_state.detach()
                with torch.no_grad():
                    expert_traj = torch.cat([demo_state, demo_action], dim=1)
                    num_inputs = self.demo_states.shape[1]
                    dist, _, _ = self.policy.forward(expert_traj[:, :num_inputs], None, None, None)
                    ac_mean = dist.mean
                    ac_std = dist.stddev
                    ac = expert_traj[:, num_inputs:]
                    ac_var = ac_std ** 2
                    lg_prob = -((ac - ac_mean) ** 2) / (2 * ac_var) - torch.log(ac_std) - math.log(math.sqrt(2 * math.pi))
                    prob = torch.exp(lg_prob.sum(-1, keepdim=True))
                    disc_val = torch.sigmoid(discrim._compute_disc_val(demo_state, demo_action))
                    eps = 1e-6
                    # clamp discriminator output to avoid div/inf
                    disc_val = disc_val.clamp(min=eps, max=1.0 - eps)
                    beta = 1.0
                    # density ratio approximation: D/(1-D) ~ p_data / p_model
                    ratio = (disc_val / (1.0 - disc_val)).clamp(min=eps, max=1e8)
                    # use (ratio - 1) to capture how much expert density exceeds model; negative -> 0
                    base = ((ratio - 1.0).clamp_min(0.0) * prob.clamp_min(eps)).pow(1.0 / (beta + 1.0))
                    expert_conf = base.clamp_min(self._expert_conf_floor)
                    expert_conf_mean = expert_conf.mean().item()

                # --------- Step 1: Inner-loop (baseline: PPO+BC) ---------
                # 目的：在 clone 上进行一次参考内环更新（非可微），得到 baseline 性能作为对照
                # 注意：这里的 actor_tmp_bc 是临时副本，不会修改真实策略参数
                actor_tmp_bc = l2l.clone_module(self.policy)
                # compute losses on temporary copy
                ac_eval_tmp = actor_tmp_bc.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )
                ratio_tmp = torch.exp(ac_eval_tmp['log_prob'] - prev_log_prob_batch)
                surr1_tmp = ratio_tmp * adv_batch
                surr2_tmp = torch.clamp(
                    ratio_tmp,
                    1.0 - self._arg('clip_param'),
                    1.0 + self._arg('clip_param')
                ) * adv_batch
                actor_loss_tmp = -torch.min(surr1_tmp, surr2_tmp).mean()
                value_pred_clipped_tmp = value_batch + (ac_eval_tmp['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param')
                )
                value_losses_tmp = (ac_eval_tmp['value'] - return_batch).pow(2)
                value_losses_clipped_tmp = (value_pred_clipped_tmp - return_batch).pow(2)
                value_loss_tmp = 0.5 * torch.max(value_losses_tmp, value_losses_clipped_tmp).mean()

                bc_loss_tmp = F.mse_loss(actor_tmp_bc.act(demo_state), demo_action)
                total_tmp_loss_bc = (
                    actor_loss_tmp +
                    self._arg('value_loss_coef') * value_loss_tmp +
                    self._arg('bc_coef') * bc_loss_tmp -
                    ac_eval_tmp['ent'].mean() * self._arg('entropy_coef')
                )

                inner_lr = 1e-1
                grads_bc = autograd.grad(
                    total_tmp_loss_bc,
                    tuple(actor_tmp_bc.parameters()),
                    create_graph=False,
                    allow_unused=True,
                )
                updates_bc = [(-inner_lr * g) if g is not None else None for g in grads_bc]
                update_module(actor_tmp_bc, updates=updates_bc)

                # evaluate baseline after inner update
                ac_eval_tmp_after = actor_tmp_bc.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )
                ratio_tmp_after = torch.exp(ac_eval_tmp_after['log_prob'] - prev_log_prob_batch)
                surr1_tmp_after = ratio_tmp_after * adv_batch
                surr2_tmp_after = torch.clamp(
                    ratio_tmp_after,
                    1.0 - self._arg('clip_param'),
                    1.0 + self._arg('clip_param')
                ) * adv_batch
                actor_loss_tmp_after = -torch.min(surr1_tmp_after, surr2_tmp_after).mean()
                value_pred_clipped_tmp_after = value_batch + (ac_eval_tmp_after['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param')
                )
                value_losses_tmp_after = (ac_eval_tmp_after['value'] - return_batch).pow(2)
                value_losses_clipped_tmp_after = (value_pred_clipped_tmp_after - return_batch).pow(2)
                value_loss_tmp_after = 0.5 * torch.max(value_losses_tmp_after, value_losses_clipped_tmp_after).mean()
                bc_loss_tmp_after = F.mse_loss(actor_tmp_bc.act(demo_state), demo_action)
                policy_bc_loss_val = (
                    actor_loss_tmp_after +
                    self._arg('value_loss_coef') * value_loss_tmp_after +
                    self._arg('bc_coef') * bc_loss_tmp_after -
                    ac_eval_tmp_after['ent'].mean() * self._arg('entropy_coef')
                )

                # --------- Step 2: Inner-loop (meta path: PPO+GILD, 可微) ---------
                # 目的：在另一个 clone(actor_tmp_gild) 上加入 GILD 指导并保留计算图（create_graph=True），
                # 以便在 Step 4 构造 meta_loss 时，梯度可以沿内环反传回 gild_network
                # 注意：只在 gild 路径保持 create_graph，可通过内存/速度权衡调整 inner_lr 或减少频率
                actor_tmp_gild = l2l.clone_module(self.policy)
                ac_eval_tmp_g = actor_tmp_gild.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )
                ratio_tmp_g = torch.exp(ac_eval_tmp_g['log_prob'] - prev_log_prob_batch)
                surr1_tmp_g = ratio_tmp_g * adv_batch
                surr2_tmp_g = torch.clamp(
                    ratio_tmp_g,
                    1.0 - self._arg('clip_param'),
                    1.0 + self._arg('clip_param')
                ) * adv_batch
                actor_loss_tmp_g = -torch.min(surr1_tmp_g, surr2_tmp_g).mean()
                value_pred_clipped_tmp_g = value_batch + (ac_eval_tmp_g['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param')
                )
                value_losses_tmp_g = (ac_eval_tmp_g['value'] - return_batch).pow(2)
                value_losses_clipped_tmp_g = (value_pred_clipped_tmp_g - return_batch).pow(2)
                value_loss_tmp_g = 0.5 * torch.max(value_losses_tmp_g, value_losses_clipped_tmp_g).mean()

                # compute gild guidance on demo data (differentiable)
                action_for_gild_tmp = actor_tmp_gild.act(demo_state)
                gild_input_tmp = torch.cat([action_for_gild_tmp, demo_state, demo_action], dim=1)
                gild_output_tmp = self.gild_network(gild_input_tmp)
                # reuse discriminator-based guidance weight
                # note: disc_val, prob computed above refer to current policy; acceptable approximation
                guidance_weight_tmp = expert_conf / expert_conf.abs().mean().detach().clamp_min(self._conf_norm_eps)
                weighted_gild_loss_raw_tmp = (gild_output_tmp.squeeze(-1) * guidance_weight_tmp.squeeze(-1)).mean()

                total_tmp_loss_gild = (
                    actor_loss_tmp_g +
                    self._arg('value_loss_coef') * value_loss_tmp_g +
                    self._arg('bc_coef') * F.mse_loss(actor_tmp_gild.act(demo_state), demo_action) -
                    ac_eval_tmp_g['ent'].mean() * self._arg('entropy_coef') +
                    self._gild_actor_coef * weighted_gild_loss_raw_tmp
                )

                grads_gild = autograd.grad(
                    total_tmp_loss_gild,
                    tuple(actor_tmp_gild.parameters()),
                    create_graph=True,
                    allow_unused=True,
                )
                updates_gild = [(-inner_lr * g) if g is not None else None for g in grads_gild]
                update_module(actor_tmp_gild, updates=updates_gild)

                # evaluate gild-updated actor
                ac_eval_tmp_g_after = actor_tmp_gild.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )
                ratio_tmp_g_after = torch.exp(ac_eval_tmp_g_after['log_prob'] - prev_log_prob_batch)
                surr1_tmp_g_after = ratio_tmp_g_after * adv_batch
                surr2_tmp_g_after = torch.clamp(
                    ratio_tmp_g_after,
                    1.0 - self._arg('clip_param'),
                    1.0 + self._arg('clip_param')
                ) * adv_batch
                actor_loss_tmp_g_after = -torch.min(surr1_tmp_g_after, surr2_tmp_g_after).mean()
                value_pred_clipped_tmp_g_after = value_batch + (ac_eval_tmp_g_after['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param')
                )
                value_losses_tmp_g_after = (ac_eval_tmp_g_after['value'] - return_batch).pow(2)
                value_losses_clipped_tmp_g_after = (value_pred_clipped_tmp_g_after - return_batch).pow(2)
                value_loss_tmp_g_after = 0.5 * torch.max(value_losses_tmp_g_after, value_losses_clipped_tmp_g_after).mean()
                bc_loss_tmp_g_after = F.mse_loss(actor_tmp_gild.act(demo_state), demo_action)
                policy_loss_val_gild = (
                    actor_loss_tmp_g_after +
                    self._arg('value_loss_coef') * value_loss_tmp_g_after +
                    self._arg('bc_coef') * bc_loss_tmp_g_after -
                    ac_eval_tmp_g_after['ent'].mean() * self._arg('entropy_coef')
                )

                # --------- Step 3: 真实策略评估并更新（PPO+GILD） ---------
                # 这里对真实策略执行实际的 actor update，影响训练参数
                ac_eval = self.policy.evaluate_actions(
                    state_batch, other_state_batch, hxs_batch, mask_batch, action_batch
                )

                ratio = torch.exp(ac_eval['log_prob'] - prev_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self._arg('clip_param'),
                    1.0 + self._arg('clip_param')
                ) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_batch + (ac_eval['value'] - value_batch).clamp(
                    -self._arg('clip_param'), self._arg('clip_param')
                )
                value_losses = (ac_eval['value'] - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                with torch.no_grad():
                    action_for_gild = self.policy.act(demo_state)

                gild_input = torch.cat([action_for_gild, demo_state, demo_action], dim=1)
                gild_output = self.gild_network(gild_input)
                guidance_weight = expert_conf / expert_conf.abs().mean().detach().clamp_min(self._conf_norm_eps)
                weighted_gild_loss_raw = (gild_output.squeeze(-1) * guidance_weight.squeeze(-1)).mean()

                # 稳健的一次性幅值校准：收集前几批的中位数后应用一次性 scale
                if not self._scale_initialized:
                    try:
                        eps_scale = 1e-8
                        current_value_term_local = (self._arg('value_loss_coef') * value_loss).detach()
                        # 收集样本（标量绝对值）备用
                        self._gild_scale_values_buf.append(float(current_value_term_local.abs().item()))
                        self._gild_scale_gilds_buf.append(float(weighted_gild_loss_raw.detach().abs().item()))
                        if len(self._gild_scale_values_buf) >= self._gild_scale_collect_batches:
                            med_val = float(np.median(np.array(self._gild_scale_values_buf)))
                            med_gild = float(np.median(np.array(self._gild_scale_gilds_buf)))
                            if med_gild < eps_scale:
                                scale = 1.0
                            else:
                                scale = med_val / (med_gild + eps_scale)
                            if not math.isfinite(scale) or scale <= 0:
                                scale = 1.0
                            scale = max(1.0, min(scale, 100.0))
                            old_coef = float(self._gild_actor_coef)
                            self._gild_actor_coef = old_coef * scale
                            self._scale_initialized = True
                            # 清理 buffer
                            self._gild_scale_values_buf = []
                            self._gild_scale_gilds_buf = []
                            print(f"[GILD Scale Calib] applied one-time median-scale={scale:.4f}, gild_actor_coef: {old_coef:.4f} -> {self._gild_actor_coef:.4f}")
                    except Exception:
                        self._scale_initialized = True

                warmup_frac = min(1.0, float(self.update_iter + 1) / max(1, self._gild_warmup_updates))
                if warmup_frac < 1.0 and not self._gild_warmup_reported:
                    print(f"[GILD Warmup Ramp] linearly ramp actor gild term over {self._gild_warmup_updates} updates")
                    self._gild_warmup_reported = True
                # 外部 gild_coef 仅保留接口，不参与当前版本的 actor 权重计算
                weighted_gild_loss = self._gild_actor_coef * warmup_frac * weighted_gild_loss_raw

                
                # print(self._arg('value_loss_coef') * value_loss)
                # print(weighted_gild_loss)
                
                
                total_actor_loss = (
                    actor_loss+
                    self._arg('value_loss_coef') * value_loss +
                    weighted_gild_loss -
                    ac_eval['ent'].mean() * self._arg('entropy_coef')
                )

                # 真实 actor 更新只影响 self.policy；Step 2 的 policy_loss_val_gild 继续保留为 meta 路径结果
                self._standard_step(total_actor_loss)  # 优化真实 actor
                self.update_iter += 1
                
                
                # --------- Step 4: Meta objective 构建与 GILD 更新 ---------
                # 逻辑：用 Step1 得到的 baseline（已 detach）与 Step2 得到的 gild 评估（可微）构造 utility
                # 并最大化 utility（这里通过最小化 -utility），保证 meta 梯度沿可微内环流回 gild_network
                # baseline 保持 detach，避免把 baseline 的梯度传回真实策略或其它不必要路径
                meta_gap = policy_bc_loss_val.detach() - policy_loss_val_gild
                utility = torch.tanh(meta_gap / self._gild_signal_temp)
                # 我们希望最大化 utility -> 最小化负 utility
                loss_meta = - utility.mean() * self._meta_signal_coef

                meta_gap_scale = (
                    policy_bc_loss_val.detach().abs() +
                    policy_loss_val_gild.abs()
                ).clamp_min(1e-6)
                utility_normalized = meta_gap / meta_gap_scale
                current_value_term = (self._arg('value_loss_coef') * value_loss).detach()
                current_disc_mean = disc_val.detach().mean().item()

                debug_step = (self.update_iter < 5) or (self.update_iter % 500 == 0)
                if debug_step:
                    print(
                        f"[GILD Debug] update={self.update_iter}, "
                        f"value_term={current_value_term.item():.4f}, "
                        f"gild_out_abs={gild_output.abs().mean().item():.4f}, "
                        f"guidance_weight={guidance_weight.abs().mean().item():.4f}, "
                        f"gild_raw={weighted_gild_loss_raw.item():.4f}, "
                        f"gild_final={weighted_gild_loss.item():.4f}, "
                        f"meta_gap={meta_gap.item():.4f}, "
                        f"utility={utility.item():.4f}, "
                        f"meta_gap_scale={meta_gap_scale.item():.4f}, "
                        f"utility_norm={utility_normalized.item():.4f}, "
                        f"meta_loss={loss_meta.item():.4f}, "
                        f"disc_mean={current_disc_mean:.4f}"
                    )

                # （已移除退火/退出逻辑，GILD 由动态目标与一次性幅值校准自然退出）

                # 更新 GILD 网络
                # 反向传播 meta loss 到 gild_network（autograd.grad 返回与 parameters 对应的梯度张量）
                self.gild_optimizer.zero_grad()
                grad_omega = autograd.grad(loss_meta, tuple(self.gild_network.parameters()), retain_graph=False, allow_unused=True)
                if debug_step:
                    grad_sq_sum = 0.0
                    grad_param_count = 0
                    for gradient in grad_omega:
                        if gradient is not None:
                            grad_sq_sum += float(gradient.pow(2).sum().item())
                            grad_param_count += 1
                    grad_norm = math.sqrt(grad_sq_sum) if grad_sq_sum > 0 else 0.0
                    print(f"[GILD Debug Grad] grad_norm={grad_norm:.6f}, grad_params={grad_param_count}")
                for gradient, variable in zip(grad_omega, self.gild_network.parameters()):
                    if gradient is None:
                        variable.grad = None
                    else:
                        variable.grad = gradient
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
                log_vals['meta_loss'] += loss_meta.item()
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
                # 清理临时变量
                del actor_tmp_bc
                del actor_tmp_gild
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
    def update(self, discrim , rollouts, args=None, beginning=False, t=1):
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