import gym
from gym import spaces
import numpy as np
from ..game.gemgem import GemBoard


class Match3Env(gym.Env):

    def __init__(self):
        self.game = GemBoard()
        self.board_width = len(self.game.gameBoard)
        assert self.board_width > 0, "棋盘宽度至少为1"
        self.board_height = len(self.game.gameBoard[0])
        assert self.board_height > 0, "棋盘高度至少为1"
        self.action_info = self.initial_action_space()

        self.action_space = spaces.Discrete(len(self.action_info))
        self.observation_space = spaces.Box(low=-1, high=200, shape=(1, self.board_width, self.board_height),
                                            dtype=np.int32)

    def initial_action_space(self):
        """
        初始化行为空间
        :return: 行为空间集合
        """

        width_action_num = (self.board_width - 1) * self.board_height
        height_action_num = self.board_width * (self.board_height - 1)
        action_space = {}
        for i in range(width_action_num + height_action_num):
            if i < width_action_num:
                y = i // (self.board_width - 1)
                x = i % (self.board_width - 1)
                first_gem = {"x": x, "y": y}
                second_gem = {"x": x + 1, "y": y}
            else:
                y = (i - width_action_num) // self.board_width
                x = (i - width_action_num) % self.board_width
                first_gem = {"x": x, "y": y}
                second_gem = {"x": x, "y": y + 1}
            action_space[i] = [first_gem, second_gem]
        return action_space

    def step(self, action):
        assert action in self.action_info, "操作违规！！"
        first_gem, second_gem = self.action_info[action]
        old_score = self.game.get_score()
        self.game.step(first_gem, second_gem)
        new_score = self.game.get_score()
        return self.get_observation(), (new_score - old_score), self.game.game_is_over(), {}

    def reset(self):
        self.game = GemBoard()
        return self.get_observation()

    def render(self, mode="human"):
        self.game.draw()

    def get_observation(self):
        game_board = np.array(self.game.get_game_board(), dtype=np.int32)
        return game_board.reshape((-1,) + game_board.shape)

    def is_over(self):
        return self.game.game_is_over()
