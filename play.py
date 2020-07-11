# from rewards import Reward
# from bot_model import tetai_model
from main import Tetris, Figure
import pygame


class InitEnviorment(Figure):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    CYAN = (0, 255, 255)

    def __init__(self):
        colors = [
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


class PlayTetris(InitEnviorment):
    def __init__(self, *args, **kwargs):

        super(PlayTetris, self).__init__(*args, **kwargs)
        self.done = None
        self.clock = None
        self.fps = None
        self.flag_set_level = None
        self.game = None
        self.counter = 0
        self.pressing_down = False

    def __enter__(self):
        self.done = False
        self.clock = pygame.time.Clock()
        self.fps = 25
        self.flag_set_level = False
        self.game = Tetris(20, 10)
        self.counter = 0
        self.pressing_down = False

    def __exit__(self, type, value, tb):
        pygame.quit()

    def __call__(self):
        done = self.done
        clock = self.clock
        fps = self.fps
        flag_set_level = self.flag_set_level
        game = self.game
        counter = self.counter
        pressing_down = self.pressing_down

        while not done:
            if game.figure is None:
                game.new_figure()
            # TODO @HIGH #discuss counter
            counter += 1
            if counter > 100000:
                counter = 0

            if counter % (fps // game.level // 2) == 0 or pressing_down:
                if game.state == "start":
                    game.go_down()

            if game.score % 1 == 0 and game.score > 0 and flag_set_level:
                if (fps // game.level // 2) == 1:
                    self.screen.blit(text_game_master, [10, 200])
                else:
                    game.level += 1
                    flag_set_level = False
                    self.screen.blit(text_game_score_update, [10, 200])
            # model_action_out = model(inputs)
            # key  = get_key(model_action_out)
            key = 0
            event = _create_event_key(key)
            _post_event(event)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        game.rotate()
                    if event.key == pygame.K_DOWN:
                        pressing_down = True
                    if event.key == pygame.K_LEFT:
                        game.go_side(-1)
                    if event.key == pygame.K_RIGHT:
                        game.go_side(1)
                    if event.key == pygame.K_SPACE:
                        game.go_space()

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        pressing_down = False

            self.screen.fill(CYAN)

            for i in range(game.height):
                for j in range(game.width):
                    pygame.draw.rect(
                        self.screen,
                        GRAY,
                        [
                            game.x + game.zoom * j,
                            game.y + game.zoom * i,
                            game.zoom,
                            game.zoom,
                        ],
                        1,
                    )
                    if game.field[i][j] > 0:
                        pygame.draw.rect(
                            self.screen,
                            self.colors[game.field[i][j]],
                            [
                                game.x + game.zoom * j + 1,
                                game.y + game.zoom * i + 1,
                                game.zoom - 2,
                                game.zoom - 1,
                            ],
                        )

            if game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in game.figure.image():
                            pygame.draw.rect(
                                self.screen,
                                self.colors[game.figure.color],
                                [
                                    game.x + game.zoom * (j + game.figure.x) + 1,
                                    game.y + game.zoom * (i + game.figure.y) + 1,
                                    game.zoom - 2,
                                    game.zoom - 2,
                                ],
                            )
            # Loop until the user clicks the close button.
            font = pygame.font.SysFont("Calibri", 25, True, False)
            font1 = pygame.font.SysFont("Calibri", 65, True, False)
            font_master = pygame.font.SysFont("Calibri", 25, True, False)
            font_update = pygame.font.SysFont("Calibri", 25, True, False)
            text = font.render("Score: " + str(game.score), True, BLACK)
            text_game_over = font1.render("Game Over :( ", True, (255, 0, 0))
            text_game_master = font_master.render("soja abbb", True, (255, 0, 0))
            text_game_score_update = font_update.render(
                "New Level Unlocked ", True, (255, 0, 0)
            )

            self.screen.blit(text, [0, 0])
            if game.state == "gameover":
                self.screen.blit(text_game_over, [10, 200])

            pygame.display.flip()
            clock.tick(fps)


KEY_MAP = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]


def _create_event_key(key):
    _key = KEY_MAP[key]
    event = pygame.event.Event(
        pygame.KEYDOWN, unicode="a", key=_key, mod=pygame.KMOD_NONE
    )  # create the event
    return event


def _post_event(event):
    pygame.event.post(event)


if __name__ == "__main__":
    play = PlayTetris()
    with play as tetris_bot:
        tetris_bot()
