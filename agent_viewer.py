# IN FORM: [maze, position list, position list, ...]
# position = [x, y, theta]

import pygame
import multiprocessing as mp
import queue
import math
import colorsys
import math
import numpy as np

# Variables
SCREEN_SIZE = (1200, 800)
FPS = 60
FRAME_TIME = 0.2
RADIUS = 0.3
SPEEDS = [0.25, 0.5, 1, 2, 4, 8]


def generate_colors(n):
    colors = []

    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1)

        colors.append(
            tuple(int(x * 255) for x in rgb)
        )

    return colors


class Dashboard:
    def __init__(self):
        self.process = None
        self.connection = None
        self.episode_number = 0

    def start(self):
        self.connection = mp.Queue()

        self.process = mp.Process(
            target=setup_viewer,
            args=(self.connection,)
        )

        self.process.start()

    def send(self, maze, positions):
        chinesedatalogger = [maze, *positions]

        self.connection.put(chinesedatalogger)
        self.episode_number += 1

    def stop(self):
        self.connection.put(None)
        self.process.join()


def setup_viewer(connection):
    pygame.init()

    screen = pygame.display.set_mode(
        SCREEN_SIZE,
        pygame.RESIZABLE
    )

    pygame.display.set_caption("Episode Viewer")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 25)

    episodes = []
    episode_colors = []

    episodenumber = 0
    timestamp = 0

    speed = 2
    playing = False
    timer = 0

    running = True

    while running:
        dt = clock.tick(FPS) / 1000
        try:
            while True:
                new_episode = connection.get_nowait()

                if new_episode is None:
                    running = False
                    break

                episodes.append(new_episode)

                number_agents = max(
                    len(frame)
                    for frame in new_episode[1:]
                )

                episode_colors.append(
                    generate_colors(number_agents)
                )

        except queue.Empty:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing

                elif event.key == pygame.K_RIGHT:
                    timestamp += 1

                elif event.key == pygame.K_LEFT:
                    timestamp -= 1

                elif event.key == pygame.K_UP:
                    episodenumber += 1
                    timestamp = 0

                elif event.key == pygame.K_DOWN:
                    episodenumber -= 1
                    timestamp = 0

                elif event.key == pygame.K_EQUALS:
                    speed += 1

                elif event.key == pygame.K_MINUS:
                    speed -= 1

        if episodes:
            episodenumber = max(
                0,
                min(episodenumber, len(episodes) - 1)
            )

            speed = max(
                0,
                min(speed, len(SPEEDS) - 1)
            )

            frames = len(episodes[episodenumber]) - 1

            timestamp = max(
                0,
                min(timestamp, frames - 1)
            )

            if playing:
                timer += dt

                if timer >= FRAME_TIME / SPEEDS[speed]:
                    timer = 0
                    timestamp += 1

                    if timestamp >= frames:
                        timestamp = frames - 1
                        playing = False

        screen.fill((220, 220, 220))

        if episodes:
            maze = episodes[episodenumber][0]

            scale = maze_generate(
                screen,
                maze
            )

            positions = episodes[episodenumber][timestamp + 1]

            position_generate(
                screen,
                positions,
                episode_colors[episodenumber],
                scale
            )

            dashboard_generate(
                screen,
                font,
                episodenumber,
                timestamp,
                len(episodes),
                len(episodes[episodenumber]) - 1,
                SPEEDS[speed],
                playing
            )

        pygame.display.flip()

    pygame.quit()


def maze_generate(screen, maze):
    scale = screen.get_height() / len(maze)

    for y in range(len(maze)):
        for x in range(len(maze[y])):

            pygame.draw.rect(
                screen,
                (0, 0, 0)
                if maze[y][x] == 1
                else (255, 255, 255),

                (
                    int(x * scale),
                    int(y * scale),
                    math.ceil(scale),
                    math.ceil(scale)
                )
            )

    return scale

def position_generate(screen, positions, colors, scale):
    overlay = pygame.Surface(
        screen.get_size(),
        pygame.SRCALPHA
    )

    radius = int(scale * RADIUS)

    for i, position in enumerate(positions):
        x, y, theta = position

        cx = int((x + 0.5) * scale)
        cy = int((y + 0.5) * scale)

        pygame.draw.circle(
            overlay,
            (*colors[i], 128),
            (cx, cy),
            radius
        )

        end_x = int(
            cx + math.cos(theta) * radius
        )

        end_y = int(
            cy - math.sin(theta) * radius
        )

        pygame.draw.line(
            overlay,
            (255, 255, 255, 255),
            (cx, cy),
            (end_x, end_y),
            3
        )

    screen.blit(overlay, (0, 0))


def dashboard_generate(
    screen,
    font,
    episode,
    timestamp,
    total_episodes,
    total_frames,
    speed,
    playing,

):
    x = screen.get_height() + 20

    text = [
        "Episode Viewer",
        "",
        f"Episode: {episode + 1}/{total_episodes}",
        f"Frame: {timestamp + 1}/{total_frames}",
        f"Speed: {speed}x",
        f"Playing: {playing}",
        "",
        "SPACE TO PLAY",
        "LEFT AND RIGHT TO CHANGE FRAME",
        "UP AND DOWN TO CHANGE EPISOE",
        "+ AND - FOR CHANGING SPEED",
    ]

    for i in range(len(text)):
        screen.blit(
            font.render(
                text[i],
                True,
                (20, 20, 20)
            ),
            (x, 30 + i * 30)
        )


maze = np.array([
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,1,0,1],
    [1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])


positions = np.array([
    # time 0
    [
        [1, 1, 0],
        [5, 5, math.pi]
    ],

])


if __name__ == "__main__":
    dash = Dashboard()
    dash.start()

    dash.send(
        maze.tolist(),
        positions.tolist()
    )

    dash.process.join()

