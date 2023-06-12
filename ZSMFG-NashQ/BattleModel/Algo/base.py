import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from magent.gridworld import GridWorld


class ValueNet(nn.Module):
    def __init__(self, env, handle, name, update_every=5, use_mf=False, learning_rate=1e-4, tau=0.005, gamma=0.95):
        super(ValueNet, self).__init__()
        self.env = env
        self.name = name

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1

        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.obs_input = nn.Parameter(torch.Tensor(*self.view_space))
        self.feat_input = nn.Parameter(torch.Tensor(*self.feature_space))
        self.mask = nn.Parameter(torch.Tensor(1))

        if self.use_mf:
            self.act_prob_input = nn.Parameter(torch.Tensor(1, self.num_actions))

        self.act_input = nn.Parameter(torch.Tensor(1))

        self._construct_net()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _construct_net(self):
        self.conv1 = nn.Conv2d(self.view_space[0], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.dense_obs = nn.Linear(np.prod(self.conv2.out_shape[1:]), 256)
        self.dense_emb = nn.Linear(self.feature_space[0], 32)
        self.concat_layer = nn.Linear(288 if self.use_mf else 256, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense_out = nn.Linear(128, 64)
        self.q_value = nn.Linear(64, self.num_actions)

    def forward(self):
        x = self.obs_input
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        flatten_obs = self.dense_obs(x)

        h_obs = nn.functional.relu(flatten_obs)
        h_emb = nn.functional.relu(self.dense_emb(self.feat_input))

        concat_layer = torch.cat([h_obs, h_emb], dim=1)

        if self.use_mf:
            prob_emb = nn.functional.relu(self.act_prob_input)
            h_act_prob = nn.functional.relu(nn.Linear(prob_emb.size(1), 32)(prob_emb))
            concat_layer = torch.cat([concat_layer, h_act_prob], dim=1)

        dense2 = nn.functional.relu(self.concat_layer(concat_layer))
        out = nn.functional.relu(self.dense2(dense2))

        q = self.q_value(out)

        return q

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        self.obs_input.data = torch.Tensor(kwargs['obs'])
        self.feat_input.data = torch.Tensor(kwargs['feature'])

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            self.act_prob_input.data = torch.Tensor(kwargs['prob'])

        t_q, e_q = self.t_q(), self.e_q()
        act_idx = np.argmax(e_q.detach().numpy(), axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]

        target_q_value = kwargs['rewards'] + (1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma

        return target_q_value.detach().numpy()

    def update(self):
        """Q-learning update"""
        self.optimizer.zero_grad()
        self.update_op()
        self.optimizer.step()

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        self.obs_input.data = torch.Tensor(kwargs['state'][0])
        self.feat_input.data = torch.Tensor(kwargs['state'][1])

        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            self.act_prob_input.data = torch.Tensor(kwargs['prob'])

        actions = self.predict().detach().numpy()
        actions = np.argmax(actions, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        self.obs_input.data = torch.Tensor(kwargs['state'][0])
        self.feat_input.data = torch.Tensor(kwargs['state'][1])
        self.target_q_input.data = torch.Tensor(kwargs['target_q'])
        self.mask.data = torch.Tensor(kwargs['masks'])

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            self.act_prob_input.data = torch.Tensor(kwargs['prob'])

        self.act_input.data = torch.Tensor(kwargs['acts'])

        self.optimizer.zero_grad()
        loss, e_q = self.train_op()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'Eval-Q': np.round(np.mean(e_q.detach().numpy()), 6),
                             'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}

    def update_op(self):
        for t_param, e_param in zip(self.t_q.parameters(), self.e_q.parameters()):
            t_param.data.copy_(self.tau * e_param.data + (1. - self.tau) * t_param.data)

    def t_q(self):
        return self._construct_net()

    def e_q(self):
        return self._construct_net()

    def train_op(self):
        e_q = self.e_q()
        e_q_max = torch.sum(torch.mul(self.act_one_hot(), e_q), dim=1)
        loss = torch.sum(torch.square(self.target_q_input - e_q_max) * self.mask) / torch.sum(self.mask)
        return loss, e_q_max

    def act_one_hot(self):
        act_one_hot = torch.zeros((self.act_input.size(0), self.num_actions))
        act_one_hot.scatter_(1, self.act_input.unsqueeze(1), 1)
        return act_one_hot


