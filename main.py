from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import Game, WriteToFile
import settings

game = Game()
recorder = WriteToFile(size=settings.SIZE, path=settings.VIDEO_PATH)

moves = {"N": (0, 1), "S": (0, -1),
         "W": (1, 0), "E": (-1, 0),
         "SW": (1, -1), "SE": (-1, -1),
         "NE": (-1, 1), "NW": (1, 1)}


def get_direction(x, y, x1, y1):
    direct = ""
    if y < y1:
        direct += "S"
    elif y > y1:
        direct += "N"

    if x < x1:
        direct += "E"
    elif x > x1:
        direct += "W"
    return moves.get(direct, (0, 1))


image, score = game.play(random.choice(list(moves.values())))

for i in range(10_000):
    game_map, x, y, r = game.get_params()
    fuels = np.argwhere(game_map == 1)
    bombs = np.argwhere(game_map == -1)
    dist = np.sum(np.abs(fuels - (x, y)), axis=1)
    x_f, y_f = fuels[np.argmin(dist)]

    move = get_direction(x, y, x_f, y_f)
    image, score = game.play(move)
    if not image:
        break
    recorder.write_to_mp4(np.array(image))
    print("Score:", score)

recorder.release()
