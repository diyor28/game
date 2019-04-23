import numpy as np
import random
import settings

import cv2

from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import load_model, Model
from keras.optimizers import Adam
from PIL import Image, ImageDraw


class Fuel:
    def __init__(self, size):
        self.size = size
        self.x, self.y = random.randint(0, self.size[0]), random.randint(0, self.size[1])
        self.r = random.randint(0, 5)

    def get_params(self):
        return self.x, self.y, self.r

    def check_fuel(self, x, y, r, game_map):
        dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        game_map[self.x - 1, self.y - 1] = 1
        if dist < self.r + r:
            game_map[self.x - 1, self.y - 1] = 0
            self.__init__(self.size)
            game_map[self.x - 1, self.y - 1] = 1
            return random.randint(0, self.r), game_map
        return 0, game_map


class Wall:
    pass


class Bomb:
    def __init__(self, size):
        self.size = size
        self.x = random.randint(0, self.size[0])
        self.y = random.randint(0, self.size[1])
        self.r = random.randint(0, 5)

    def get_params(self):
        return self.x, self.y, self.r

    def check_bomb(self, x, y, r, map_):
        dist = (self.x - x) ** 2 + (self.y - y) ** 2
        map_[self.x - 1, self.y - 1] = -1
        if dist < (self.r + r) ** 2:
            map_[self.x - 1, self.y - 1] = 0
            self.__init__(self.size)
            map_[self.x - 1, self.y - 1] = -1
            return random.randint(0, self.r), map_
        return 0, map_


class Game:
    def __init__(self):
        self.size = settings.SIZE
        self.fuels = [Fuel(self.size) for i in range(20)]
        self.bombs = [Bomb(self.size) for i in range(10)]
        self.game_map = np.zeros(self.size)
        self.x, self.y = random.randint(0, self.size[0]), random.randint(0, self.size[1])
        self.r = 10
        self.score = 0
        self.moves = 10_000
        self.possible_actions = [(0, 1), (0, -1),
                                 (1, 0), (-1, 0),
                                 (1, -1), (-1, -1),
                                 (-1, 1), (1, 1)]

    def get_params(self):
        return self.game_map, self.x, self.y, self.r

    def play(self, move):
        if move not in self.possible_actions:
            raise Exception(f"Given action doesn't exist {move}")

        self.moves -= 1

        """
        move: 
        (0, 1) - North 
        (0, -1) - South 
        (1, 0) - West 
        (-1, 0) - East
        (1, -1) - South-West 
        (-1, -1) - South East
        (-1, 1) - North-East
        (1, 1) - North-West

        """
        if self.moves < 1:
            self.__init__()
            return None, self.score

        x, y = move

        if self.r <= self.x <= self.size[0] - self.r and self.r <= self.y <= self.size[1] - self.r:
            self.x -= x * 2
            self.y -= y * 2
        else:
            self.x = random.randint(0, self.size[0])
            self.y = random.randint(0, self.size[1])

        return self.animate(), self.score

    def animate(self):
        image = Image.new('RGB', self.size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r),
                     fill=(0, 200, 124))

        for fuel in self.fuels:
            x, y, r = fuel.get_params()
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            score, self.game_map = fuel.check_fuel(self.x, self.y, self.r, self.game_map)
            self.score += score

        for bomb in self.bombs:
            x, y, r = bomb.get_params()
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(40, 30, 0))
            score, self.game_map = bomb.check_bomb(self.x, self.y, self.r, self.game_map)
            self.score -= score
        return image


"""
Game consists of a player (green), fuels (red) and bombs (dark blue)
If the player eats a fuel he gains points and the opposite for bombs, he losses points.
The goal is to maximize the total score by the end of the round.
Images are solely for visualization purposes. 

One round is 10000 moves long.

Each one represents a move in a certain direction.
(0, 1) - North 
(0, -1) - South 
(1, 0) - West 
(-1, 0) - East
(1, -1) - South-West 
(-1, -1) - South East
(-1, 1) - North-East
(1, 1) - North-West


get_params() method allows to access map, x and y coordinates and radius.
Map is a (400, 400) matrix of zeros.
Use map to build a strategy. 0 is an empty square, 1 is a fuel and -1 is a bomb. Your location isn't marked on the map

Method play() controls the player. Use of private class atributes is not allowed
"""


class WriteToFile:
    def __init__(self, size, path, frame_rate=30):
        self.frames = 0
        self.size = size  # size = frame.shape[:2][::-1]
        self.writer = cv2.VideoWriter(path, 0x00000021, frame_rate, size)

    def write_to_mp4(self, frame):
        if frame.shape[:2][::-1] != self.size:
            cv2.resize(frame, dsize=self.size)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
        if settings.DEBUG:
            cv2.imshow('Frame', frame)

    def release(self):  # please call at the end
        self.writer.release()
        cv2.destroyAllWindows()


class Agent:
    def __init__(self, load_weights=False):
        self.input_size = settings.SIZE
        self.model = self.build_model()
        self.discount = settings.GAMMA
        self.output_size = 8
        self.frames_cnt = 0
        self.max_memory = 2000
        self.memory = []
        if load_weights:
            self.model.load_weights(settings.WEIGHTS)

    def build_model(self):
        input_layer = Input(self.input_size + (3,))
        conv = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
        conv = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(conv)
        flatten = Flatten()(conv)
        dense = Dense(128, activation='relu')(flatten)
        output = Dense(self.output_size, activation='linear')(dense)
        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(Adam(lr=settings.LEARNING_RATE), loss='mse')
        return model

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def act(self, game_screen):
        self.model.predict(game_screen)
        return

    def predict(self, state):
        return self.model.predict(state)

    def train(self, batch_size):
        len_memory = len(self.memory)
        if len_memory < batch_size:
            return

        inputs = np.zeros((batch_size, ) + self.input_size)
        targets = np.zeros((batch_size, self.output_size))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=batch_size)):

            state, action, reward, new_state = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state
            targets[i:i+1] = self.predict(state)

            q_sa = np.max(self.predict(new_state))

            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * q_sa

        self.model.train_on_batch(inputs, targets)


    def epsilon(self):
        if self.frames_cnt >= settings.EPS_STEPS:
            return settings.EPS_STOP
        else:
            decay = self.frames_cnt * (settings.EPS_STOP - settings.EPS_START) / settings.EPS_STEPS
            return settings.EPS_START + decay
