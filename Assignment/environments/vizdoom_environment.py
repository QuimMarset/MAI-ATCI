from vizdoom.vizdoom import DoomGame, ScreenFormat
import numpy as np
from collections import deque
import cv2
from constants import VIZDOOM_CONFIGS_PATH
import os


class VizDoomEnvironment:

    def __init__(self, env_name, num_stacked=4, num_skipped=4, frame_resize=(100, 100), reward_scale=1, render=False):
        self.reward_scale = reward_scale
        self.frame_stack = deque([], maxlen=num_stacked)
        self.frame_resize = frame_resize
        self.state_shape = (*frame_resize, num_stacked)
        self.num_skipped = num_skipped
        self.create_game(env_name, render)
        self.num_actions = self.game.get_available_buttons_size()
        self.actions = np.identity(self.num_actions, dtype = int).tolist()


    def create_game(self, env_name, render):
        config_file_path = os.path.join(VIZDOOM_CONFIGS_PATH, f'{env_name}.cfg')
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_window_visible(render)
        self.game.set_sound_enabled(False)
        self.game.set_render_hud(False)
        self.game.init()


    def process_frame(self, frame):
        frame = frame / 255.0
        frame = cv2.resize(frame, self.frame_resize)
        return frame


    def init_deque(self, frame):
        self.frame_stack.clear()
        frame = self.process_frame(frame)
        for _ in range(self.frame_stack.maxlen):
            self.frame_stack.append(frame)


    def stack_frame(self, frame):
        frame = self.process_frame(frame)
        self.frame_stack.append(frame)


    def get_stacked_frames(self):
        return np.stack(self.frame_stack, axis=2)


    def start(self):
        self.game.new_episode()
        frame = self.game.get_state().screen_buffer
        self.init_deque(frame)
        return self.get_stacked_frames()


    def step(self, action):
        reward = self.game.make_action(self.actions[action], self.num_skipped)
        terminal = self.game.is_episode_finished()
        
        if terminal:
            next_state = np.zeros(self.state_shape)
        else:
            next_frame = self.game.get_state().screen_buffer
            self.stack_frame(next_frame)
            next_state = self.get_stacked_frames()

        return next_state, reward*self.reward_scale, terminal


    def get_state_shape(self):
        return self.state_shape

    
    def get_action_space(self):
        # The function has this name to match with Gym environments
        return self.num_actions


    def end(self):
        self.game.close()