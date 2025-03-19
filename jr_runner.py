import numpy as np

from SpiderSolitaireEnv import SpiderSolitaireEnv

# Initialize environment
env = SpiderSolitaireEnv(suits=4, seed=42)

# Reset the environment (Start a new game)
state = env.reset()
print("\n--- Initial Game State ---")
print(state)

# Run a few steps with random valid actions
for step in range(5):
    print(f"\n--- Step {step + 1} ---")

    # Get all possible moves
    possible_moves = env.game.get_possible_moves()

    if not possible_moves:
        print("No valid moves available.")
        break
    if step >= 2:
        # Pick a random move
        action = np.random.randint(0, len(possible_moves))
    elif step == 0:
        action = 0
    else:
        action = -1
    observation, reward, done, _ = env.step(action)

    print(f"Action Taken: {possible_moves[action]}")
    print(f"Reward: {reward}")
    print("Updated Game State:")
    print(observation)

    if done:
        print("Game Over!")
        break

# Close the environment
env.close()
print("Environment closed.")
