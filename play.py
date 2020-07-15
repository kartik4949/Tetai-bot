import time

import pandas as pd
import pygame
import tensorflow as tf
import numpy as np
from rewards import Rewards
from bot_model import tetai_model as TetaiBrain
from main import Tetris, Figure

class InitEnvironment(Figure):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    CYAN = (0, 255, 255)

    def __init__(self):
        self.colors = [
            (0, 0, 0),
            (120, 37, 179),
            (100, 179, 179),
            (80, 34, 22),
            (80, 134, 22),
            (180, 34, 22),
            (180, 34, 122),
        ]
        # Initialize the game engine
        pygame.init()

        # Define some colors
        _size = (450, 600)
        self.screen = pygame.display.set_mode(_size)

        pygame.display.set_caption("Tetris")


class TetrisEngine(InitEnvironment):
    def __init__(self, *args, **kwargs):

        super(TetrisEngine, self).__init__(*args, **kwargs)
        self._init_environment = super(TetrisEngine, self)
        self.BLACK = self._init_environment.BLACK
        self.WHITE = self._init_environment.WHITE
        self.GRAY = self._init_environment.GRAY
        self.CYAN = self._init_environment.CYAN
        self.clock = pygame.time.Clock()
        self.fps = 25
        self.flag_set_level = False
        self.game = Tetris(20, 10)
        self.counter = 0
        self.pressing_down = False



    def __enter__(self):
        self.clock = pygame.time.Clock()
        self.fps = 25
        self.flag_set_level = False
        self.game = Tetris(20, 10)
        self.counter = 0
        self.pressing_down = False
        return self

    def restart(self):
        self.game.state = "start"
        _field = np.asarray(self.game.field)
        _field = np.zeros(_field.shape)
        self.game.field = [list(field) for field in _field]

    def quit(self):
        pygame.display.quit()

    def __call__(self ):
        done = False
        _lines_popped = 0

        if self.game.figure is None:
            self.game.new_figure()
        # TODO @HIGH #discuss self.counter
        self.counter += 1
        if self.counter > 100000:
            self.counter = 0

        if self.counter % (self.fps // self.game.level // 2) == 0 or self.pressing_down:
            if self.game.state == "start":
                _lines_popped = self.game.go_down()

        if self.game.score % 1 == 0 and self.game.score > 0 and self.flag_set_level:
            if (self.fps // self.game.level // 2) == 1:
                pass
                #self.screen.blit(text_game_master, [10, 200])
            else:
                self.game.level += 1
                self.flag_set_level = False
                self.screen.blit(text_game_score_update, [10, 200])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.game.rotate()
                if event.key == pygame.K_DOWN:
                    self.pressing_down = True
                if event.key == pygame.K_LEFT:
                    self.game.go_side(-1)
                if event.key == pygame.K_RIGHT:
                    self.game.go_side(1)
                if event.key == pygame.K_SPACE:
                    _lines_popped = self.game.go_space()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    self.pressing_down = False

        self.screen.fill(self.CYAN)

        for i in range(self.game.height):
            for j in range(self.game.width):
                pygame.draw.rect(
                    self.screen,
                    self.GRAY,
                    [
                        self.game.x + self.game.zoom * j,
                        self.game.y + self.game.zoom * i,
                        self.game.zoom,
                        self.game.zoom,
                    ],
                    1,
                )
                if self.game.field[i][j] > 0:
                    pygame.draw.rect(
                        self.screen,
                        self.colors[self.game.field[i][j]],
                        [
                            self.game.x + self.game.zoom * j + 1,
                            self.game.y + self.game.zoom * i + 1,
                            self.game.zoom - 2,
                            self.game.zoom - 1,
                        ],
                    )

        if self.game.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.game.figure.image():
                        pygame.draw.rect(
                            self.screen,
                            self.colors[self.game.figure.color],
                            [
                                self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                                self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                                self.game.zoom - 2,
                                self.game.zoom - 2,
                            ],
                        )
        # Loop until the user clicks the close button.
        font = pygame.font.SysFont("Calibri", 25, True, False)
        font1 = pygame.font.SysFont("Calibri", 65, True, False)
        font_master = pygame.font.SysFont("Calibri", 25, True, False)
        font_update = pygame.font.SysFont("Calibri", 25, True, False)
        text = font.render("Score: " + str(self.game.score), True, self.BLACK)
        text_game_over = font1.render("Game Over :( ", True, (255, 0, 0))
        text_game_master = font_master.render("soja abbb", True, (255, 0, 0))
        text_game_score_update = font_update.render(
            "New Level Unlocked ", True, (255, 0, 0)
        )

        self.screen.blit(text, [0, 0])
        if self.game.state == "gameover":
            done = True
            #self.screen.blit(text_game_over, [10, 200])

        pygame.display.flip()
        self.clock.tick(self.fps)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return done ,_lines_popped , image_data , self.game.field


class Play(TetrisEngine):
    KEY_MAP = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]

    def __init__(self):
        super(Play ,self).__init__()
        self.tetris_bot = super(Play ,self).__call__
        self.brain = TetaiBrain()
        self.reward = Rewards("MaxHeight")

    @staticmethod
    def _create_event_key(key):
        _key = Play.KEY_MAP[key]
        event = pygame.event.Event(
            pygame.KEYDOWN, unicode="_", key=_key, mod=pygame.KMOD_NONE
        )  # create the event
        return event

    @staticmethod
    def _post_event(event):
        pygame.event.post(event)

    def get_action(self, inputs):
        key = self.brain(inputs).numpy()[0]
        # NN inference
        return key

    @staticmethod
    def _calc_reponse_time(level):
        return 1/level

    def frame_step(self, key):
        _event = self.__class__._create_event_key(key)
        self.__class__._post_event(_event)
        done ,_lines_popped , frame, field= self.tetris_bot()
        reward = self.reward.reward(field, _lines_popped)
        if done:
            return frame , -1.0 , 1
        return frame , reward ,  int(done)

    def __call__(self):

        with TetrisEngine() as tetris_bot:

            done = False
            level = 1
            while not done:
                inputs = tf.random.uniform((1, 20, 10))
                _key = self.get_action(inputs)
                _response_time = self.__class__._calc_reponse_time(level)
                time.sleep(1/25)
                _event = self.__class__._create_event_key(_key)
                self.__class__._post_event(_event)
                done ,reward , image_data = tetris_bot(done)



if __name__ == "__main__":
    game = Play()

    for _ in range(100):
        _out = game.frame_step(3)
        print(_out[1])
    breakpoint()
    # print(out)
