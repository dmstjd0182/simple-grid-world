import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, agent_location: list[int, int] = None, target_location: list[list[int, int]] = 1, render_mode=None, row=5, col=5, term_reward=1, step_reward=0):
        self.row = row      # The row size of the grid
        self.col = col      # The column size of the grid
        self.window_row_size = 1024  # The size of the PyGame window
        self.window_col_size = int(self.window_row_size * self.col / self.row)
        
        # random or init agent
        self.init_agent = np.array(agent_location)
        # random target
        if type(target_location) == int:
            target = []
            while len(target) < target_location:
                new = np.random.randint(low=np.array([0, 0]), high=np.array([self.row, self.col]), size=2, dtype=int)
                # target 중복 없고 agent와 겹치지 X
                if all(((new!=t).any() for t in target)) and (self.init_agent != new).any():
                    target.append(new)
            self.init_target = np.array(target)
        # initial target
        else:
            self.init_target = np.array(target_location)

        self.term_reward = term_reward
        self.step_reward = step_reward

        # Observations are tuple with the agent's and the target's location.
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=np.array([0, 0]), high=np.array([row - 1, col - 1]), shape=(2,), dtype=int),
                spaces.Sequence(spaces.Box(low=np.array([0, 0]), high=np.array([row - 1, col - 1]), shape=(2,), dtype=int)),
            )
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return (self.agent_location, self.target_location)
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set initial location of agent and target
        self.agent_location = (
            np.random.randint(low=np.array([0, 0]), high=np.array([self.row, self.col]), size=2, dtype=int)
            if self.init_agent == None
            else self.init_agent
        )
        self.target_location = self.init_target

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0, 1, 2, 3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self.agent_location = np.clip(
            self.agent_location + direction, np.array([0, 0]), np.array([self.row - 1, self.col - 1])
        )
        # An episode is done if the agent has reached the target
        terminated = False
        for target_location in self.target_location:
            terminated = np.array_equal(self.agent_location, target_location)
            if terminated: 
                break
        reward = self.term_reward if terminated else self.step_reward   # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_col_size, self.window_row_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_col_size, self.window_row_size))
        canvas.fill((255, 255, 255))
        pix_square_row_size = (
            self.window_row_size / self.row
        )   # The row size of a single grid square in pixels
        pix_square_col_size = (
            self.window_col_size / self.col
        )   # The col size of a single grid square in pixels

        # First we draw the target
        for target_location in self.target_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    np.array([pix_square_col_size, pix_square_row_size]) * target_location[::-1],
                    (pix_square_col_size, pix_square_row_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_location[::-1] + 0.5) * np.array([pix_square_col_size, pix_square_row_size]),
            min(pix_square_col_size, pix_square_row_size) / 3,
        )

        # Finally, add some horizontal gridlines
        for x in range(self.row + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_col_size * x),
                (self.window_col_size, pix_square_col_size * x),
                width=2,
            )
        # Add some vertical gridlines
        for x in range(self.col + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_row_size * x, 0),
                (pix_square_row_size * x, self.window_row_size),
                width=2,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()