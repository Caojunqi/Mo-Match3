from typing import Callable, Dict, List, Optional, Type, Union

import gym
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from torch import nn


class Match3FeaturesExtractor(nn.Module):
    def __init__(self, observation_space):
        super(Match3FeaturesExtractor, self).__init__()

        self.features_dim = 3

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

    def forward(self, state_input):
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Match3MlpExtractor(nn.Module):

    def __init__(self, board_width, board_height):
        super(Match3MlpExtractor, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.action_num = (board_width - 1) * board_height + board_width * (board_height - 1)

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.action_num
        self.latent_dim_vf = 1

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 1, kernel_size=(1, 1))
        self.act_fc1 = nn.Linear(self.board_width * self.board_height, self.action_num)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=(1, 1))
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # action policy layers
        x_act = self.forward_actor(state_input)
        # state value layers
        x_val = self.forward_critic(state_input)
        return x_act, x_val

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x_act = th.relu(self.act_conv1(features))
        x_act = x_act.view(-1, self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        return x_act

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x_val = th.relu(self.val_conv1(features))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = th.relu(self.val_fc1(x_val))
        x_val = th.tanh(self.val_fc2(x_val))
        return x_val


class Match3Policy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(Match3Policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        board_width = self.observation_space.shape[1]
        board_height = self.observation_space.shape[2]
        self.mlp_extractor = Match3MlpExtractor(board_width, board_height)


class FeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
