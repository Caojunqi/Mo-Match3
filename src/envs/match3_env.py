import gym
from gym import error, spaces, utils
from gym.utils import seeding

from src.envs.game import Game, Point
from src.envs.game import OutOfBoardError, ImmovableShapeError
from src.envs.levels import LEVELS, Match3Levels
from src.envs.renderer import Renderer

from itertools import product
import warnings
import numpy as np

BOARD_NDIM = 2


class Match3Env(gym.Env):
    metadata = {'render.modes': None}

    def __init__(self, rollout_len=100, all_moves=False, levels=None, random_state=None):
        self.rollout_len = rollout_len
        self.random_state = random_state
        self.all_moves = all_moves
        self.levels = levels or Match3Levels(LEVELS)
        self.h = self.levels.h
        self.w = self.levels.w
        self.n_shapes = self.levels.n_shapes
        self.__episode_counter = 0

        self.__game = Game(
            rows=self.h,
            columns=self.w,
            n_shapes=self.n_shapes,
            length=3,
            all_moves=all_moves,
            random_state=self.random_state)
        self.reset()
        self.renderer = Renderer(self.n_shapes)

        # setting observation space
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=self.n_shapes,
        #     shape=(1, *self.__game.board.board_size),
        #     dtype=int)
        self.observation_space = spaces.Box(
            low=0,
            high=self.n_shapes,
            shape=(self.n_shapes + 1, *self.__game.board.board_size),
            dtype=int)

        # setting actions space
        self.__match3_actions = self.__get_available_actions()
        self.action_space = spaces.Discrete(
            len(self.__match3_actions))

    @staticmethod
    def __get_directions(board_ndim):
        """ get available directions for any number of dimensions """
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    def __points_generator(self):
        """ iterates over points on the board """
        rows, cols = self.__game.board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def __get_available_actions(self):
        """ calculate available actions for current board sizes """
        actions = set()
        directions = self.__get_directions(board_ndim=BOARD_NDIM)
        for point in self.__points_generator():
            for axis_dirs in directions:
                for dir_ in axis_dirs:
                    dir_p = Point(*dir_)
                    new_point = point + dir_p
                    try:
                        _ = self.__game.board[new_point]
                        actions.add(frozenset((point, new_point)))
                    except OutOfBoardError:
                        continue
        return list(actions)

    def __get_action(self, ind):
        return self.__match3_actions[ind]

    def step(self, action):
        # make action
        m3_action = self.__get_action(action)
        reward = self.__swap(*m3_action)

        # change counter even action wasn't successful
        self.__episode_counter += 1
        if self.__episode_counter >= self.rollout_len:
            episode_over = True
            self.__episode_counter = 0
            ob = self.reset()
        else:
            episode_over = False
            ob = self.__get_board()

        return ob, reward, episode_over, {}

    def reset(self, *args, **kwargs):
        np.random.seed(self.random_state)
        board = self.levels.sample()
        self.__game.start(board)
        return self.__get_board()

    def __swap(self, point1, point2):
        try:
            reward = self.__game.swap(point1, point2)
        except ImmovableShapeError:
            reward = 0
        return reward

    def __get_board(self):
        multi_channel_board = np.zeros(shape=(self.n_shapes + 1, *self.__game.board.board_size))
        original_board = self.__game.board.board
        for i in range(self.h):
            for j in range(self.w):
                tile_type = int(original_board[i][j])
                if tile_type == -1:
                    multi_channel_board[-1][i][j] = 1
                else:
                    multi_channel_board[tile_type][i][j] = 1
        return multi_channel_board
        # return np.expand_dims(self.__game.board.board.copy(), axis=0)

    def render(self, mode='human', close=False):
        if close:
            warnings.warn("close=True isn't supported yet")
        self.renderer.render_board(self.__game.board)

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((len(self.__match3_actions),))
        for i in range(len(self.__match3_actions)):
            first, second = self.__match3_actions[i]
            try:
                if self.__game.check_matches(first, second):
                    masks[i] = True
            except (OutOfBoardError, ImmovableShapeError):
                continue
        return masks

    def get_action(self, ind):
        return self.__match3_actions[ind]
