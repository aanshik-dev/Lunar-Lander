import gymnasium as gym
import pygame
import numpy as np

# Initialize Pygame and create a window.
pygame.init()
window_width, window_height = 800, 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Lunar Lander - Human Play")

# Define key-to-action mapping:
# 'w' -> Fire main engine (action 2)
# 'a' -> Fire left engine (action 1)
# 'd' -> Fire right engine (action 3)
# 's' -> Do nothing (action 0)
key_to_action = {
    pygame.K_w: 2,
    pygame.K_a: 1,
    pygame.K_d: 3,
    pygame.K_s: 0
}

def get_action():
    """Return the action corresponding to the currently pressed key(s).
       Defaults to 0 (do nothing) if none of the mapped keys is pressed."""
    keys = pygame.key.get_pressed()
    for key, action in key_to_action.items():
        if keys[key]:
            return action
    return 0

# Create the LunarLander-v3 environment with "rgb_array" render_mode.
env = gym.make("LunarLander-v3", render_mode="rgb_array")
clock = pygame.time.Clock()
fps = 30

running = True
while running:
    observation, info = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        # Process quit events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True

        # Get action based on pressed keys.
        action = get_action()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Render the frame as an RGB array and convert it to a Pygame surface.
        frame = env.render()  # rgb_array output
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        # Scale the frame to fill the window.
        frame_surface = pygame.transform.scale(frame_surface, (window_width, window_height))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    # Episode ended: print the obtained score.
    print(f"Episode score: {episode_reward:.2f}")

    # Wait for user input to restart or quit.
    waiting = True
    print("Press any key to play another episode (press 'Q' to quit)...")
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                waiting = False

env.close()
pygame.quit()
