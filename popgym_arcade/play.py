#!/usr/bin/env python3

import argparse

import jax
import numpy as np
import pygame

import popgym_arcade
import popgym_arcade.registration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", nargs="?", help="Launch an environment directly, e.g. BattleShipEasy"
    )
    parser.add_argument(
        "--partial", "-p", help="Play the POMDP variant, else MDP", action="store_true"
    )
    parser.add_argument(
        "--obs-size",
        "-o",
        help="Pixel size of observations, can be 128 or 256",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    if args.env is not None and args.env not in (
        popgym_arcade.registration.REGISTERED_ENVIRONMENTS
    ):
        parser.error(f"Unknown environment: {args.env}")
    assert args.obs_size in [
        128,
        256,
    ], f"Invalid obs-size: {args.obs_size}, must be in [128, 256]"
    return args


def to_surf(arr):
    # Convert jax arry to pygame surface
    return np.transpose(arr, (1, 0, 2))


def play(args):
    # Create env env variant
    env, env_params = popgym_arcade.make(
        args.env, partial_obs=args.partial, obs_size=args.obs_size
    )

    # Vectorize and compile the env
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)

    # Initialize environment
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    observation, env_state = env_reset(reset_key, env_params)

    screen = pygame.display.set_mode((args.obs_size, args.obs_size))
    pygame.display.set_caption(args.env)
    clock = pygame.time.Clock()

    # Convert numpy array to Pygame surface
    surface = pygame.surfarray.make_surface(to_surf(observation))

    # Action mappings (modify based on your environment's action space)
    ACTION_MEANINGS = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_SPACE: 4,
        # Add more keys if needed
    }
    print("Controls: up, down, left, right, spacebar")

    while True:
        # Handle events
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                if event.key in ACTION_MEANINGS:
                    action = ACTION_MEANINGS[event.key]

        # Render to screen
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Take action if key pressed
        if action is not None:
            key, step_key = jax.random.split(key)
            observation, env_state, reward, done, info = env_step(
                step_key, env_state, action, env_params
            )
            surface = pygame.surfarray.make_surface(to_surf(observation))

            # Render to screen
            if done:
                print("Game over")
                break

        clock.tick(30)  # Control FPS

    return True


def menu(args):
    difficulties = ("Easy", "Medium", "Hard")
    games = [
        env.removesuffix("Easy")
        for env in popgym_arcade.registration.REGISTERED_ENVIRONMENTS
        if env.endswith("Easy")
    ]
    game = games[0]
    difficulty = difficulties[0]
    partial = args.partial
    font = pygame.font.Font(None, 24)
    title = pygame.font.Font(None, 42)
    clock = pygame.time.Clock()

    def open_menu():
        pygame.display.set_caption("POPGym Arcade")
        return pygame.display.set_mode((560, 410))

    screen = open_menu()
    while True:
        screen.fill("#151c29")
        screen.blit(title.render("POPGym Arcade", True, "#f5f7ff"), (24, 20))

        def button(label, rect, selected=False):
            pygame.draw.rect(
                screen, "#3863a2" if selected else "#273448", rect, border_radius=7
            )
            surface = font.render(label, True, "#f5f7ff")
            screen.blit(surface, surface.get_rect(center=rect.center))

        game_buttons = {}
        for index, name in enumerate(games):
            row, column = divmod(index, 2)
            rect = pygame.Rect(24 + column * 260, 68 + row * 38, 252, 32)
            game_buttons[name] = rect
            button(name, rect, name == game)

        difficulty_buttons = {}
        for index, name in enumerate(difficulties):
            rect = pygame.Rect(24 + index * 104, 270, 96, 34)
            difficulty_buttons[name] = rect
            button(name, rect, name == difficulty)
        mode_button = pygame.Rect(336, 270, 200, 34)
        button("Partial observation" if partial else "Full observation", mode_button)
        play_button = pygame.Rect(408, 322, 128, 36)
        button("Play", play_button, True)
        screen.blit(
            font.render("Arrows: game   D: difficulty   M: observation", True, "#b9c7df"),
            (24, 332),
        )
        screen.blit(
            font.render("Enter: play   •   Esc: quit arcade", True, "#b9c7df"),
            (24, 374),
        )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
            launch = event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN
            if event.type == pygame.KEYDOWN:
                moves = {
                    pygame.K_LEFT: -1,
                    pygame.K_RIGHT: 1,
                    pygame.K_UP: -2,
                    pygame.K_DOWN: 2,
                }
                if event.key in moves:
                    game = games[(games.index(game) + moves[event.key]) % len(games)]
                elif event.key == pygame.K_d:
                    difficulty = difficulties[
                        (difficulties.index(difficulty) + 1) % len(difficulties)
                    ]
                elif event.key == pygame.K_m:
                    partial = not partial
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for name, rect in game_buttons.items():
                    if rect.collidepoint(event.pos):
                        game = name
                for name, rect in difficulty_buttons.items():
                    if rect.collidepoint(event.pos):
                        difficulty = name
                if mode_button.collidepoint(event.pos):
                    partial = not partial
                launch = play_button.collidepoint(event.pos)
            if launch:
                selection = argparse.Namespace(
                    env=game + difficulty, partial=partial, obs_size=args.obs_size
                )
                if not play(selection):
                    return
                screen = open_menu()
                break
        clock.tick(30)


def main():
    args = parse_args()
    pygame.display.init()
    pygame.font.init()
    try:
        if args.env is None:
            menu(args)
        else:
            play(args)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
