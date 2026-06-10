import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from rlf.policies.base_policy import ActionData
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.actor_critic.base_actor_critic import ActorCritic


class DistActorCritic(ActorCritic):
    """
    Defines an actor/critic where the actor outputs an action distribution
    """

    def __init__(self,
                 get_actor_fn=None,
                 get_dist_fn=None,
                 get_critic_fn=None,
                 get_critic_head_fn=None,
                 fuse_states=[],
                 use_goal=False,
                 get_base_net_fn=None):
        super().__init__(get_critic_fn, get_critic_head_fn, use_goal,
                fuse_states, get_base_net_fn)
        """
        - get_actor_fn: (obs_space : (int), input_shape : (int) ->
          rlf.rl.model.BaseNet)
        """

        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        self.get_actor_fn = get_actor_fn

        if get_dist_fn is None:
            get_dist_fn = putils.get_def_dist
        self.get_dist_fn = get_dist_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.actor = self.get_actor_fn(
            rutils.get_obs_shape(obs_space, args.policy_ob_key),
            self._get_base_out_shape())
        self.dist = self.get_dist_fn(
            self.actor.output_shape, self.action_space)

    def get_action(self, state, add_state, hxs, masks, step_info):
        dist, value, hxs = self.forward(state, add_state, hxs, masks)
        if self.args.deterministic_policy:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return ActionData(value, action, action_log_probs, hxs, {
            'dist_entropy': dist_entropy
        })

    def forward(self, state, add_state, hxs, masks):
        if masks is None:
            masks = torch.ones(state.size(0), 1, device=state.device)  # 默认全 1 的 mask
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)

        value = self._get_value_from_features(base_features, hxs, masks)

        actor_features, _ = self.actor(base_features, hxs, masks)

        dist = self.dist(actor_features)

        return dist, value, hxs

    def evaluate_actions(self, state, add_state, hxs, masks, action):
        #import ipdb; ipdb.set_trace()
        dist, value, hxs = self.forward(state, add_state, hxs, masks)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return {
            'value': value,
            'log_prob': action_log_probs,
            'ent': dist_entropy,
        }

    # def evaluate_actions(self, state, add_state, hxs, masks, action, modulation=None):
    #     dist, value, hxs = self.forward(state, add_state, hxs, masks)
    
    #     if modulation is not None:
    #         if hasattr(dist, 'mean'):
    #             new_mean = dist.mean + modulation
    #             dist = torch.distributions.Normal(new_mean, dist.stddev)  # 假设是 Normal 分布
    
    #     action_log_probs = dist.log_probs(action)
    #     dist_entropy = dist.entropy()
    #     return {
    #         'value': value,
    #         'log_prob': action_log_probs,
    #         'ent': dist_entropy,
    #     }




    def get_actor_params(self):
        return super().get_actor_params() + list(self.dist.parameters())

    # def act(self, state):
    #     """
    #     A simplified method to get actions directly from the state.
    #     """
    #     action_data = self.get_action(state, None, None, None, None)
    #     return action_data.action
    
    def act(self, state):
        """
        A simplified method to get actions directly from the state.
        """
        dist, _, _ = self.forward(state, None, None, None)
        action = dist.rsample()  # 使用 rsample 保留计算图
        return action

    def get_value_for_meta(self, state, action, add_state=None, hxs=None, masks=None):
        """
        用于Meta-Learning部分，不影响原本get_value。
        输入state和action，内部提取特征然后跑critic。
        """
        if masks is None:
            # 默认值为全 1 的张量，表示所有状态都有效
            masks = torch.ones(state.size(0), 1, device=state.device)

        base_features, _ = self._apply_base_net(state, add_state, hxs, masks)
        value = self.critic(base_features, action, masks)
        return value
    
