# Filename: train_agent.py
import gymnasium as gym
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

# Constants for the Neural Network Architecture
INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
GENE_SIZE = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

def policy_action(params, observation):
    W1 = params[0:128].reshape(8, 16)
    b1 = params[128:144].reshape(16)
    W2 = params[144:208].reshape(16, 4)
    b2 = params[208:212].reshape(4)
    hidden = np.maximum(0, np.dot(observation, W1) + b1)
    logits = np.dot(hidden, W2) + b2
    return np.argmax(logits)

def evaluate_individual(params):
    """Function to be run in parallel."""
    episodes = 5
    total_reward = 0.0
    env = gym.make('LunarLander-v3')
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    env.close()
    return total_reward / episodes

def train_and_save(filename, population_size=64, generations=100):
    print(f"Starting training with Population: {population_size}, Gens: {generations}")
    
    # Initialize mean and standard deviation
    mean = np.random.randn(GENE_SIZE) * 0.1
    sigma = 0.5
    best_reward = -np.inf
    best_params = mean.copy()

    for gen in range(generations):
        # Generate population (Noise Table approach)
        noise = np.random.randn(population_size, GENE_SIZE)
        candidates = mean + sigma * noise
        
        # Parallel Evaluation
        with ProcessPoolExecutor() as executor:
            fitness = list(executor.map(evaluate_individual, candidates))
        
        fitness = np.array(fitness)
        
        # Rank-based weight calculation (Evolution Strategy)
        ranks = np.argsort(np.argsort(fitness)) 
        utilities = np.maximum(0, np.log(population_size / 2 + 1) - np.log(population_size - ranks))
        utilities /= np.sum(utilities)
        
        # Update mean
        old_mean = mean.copy()
        mean += np.dot(utilities, candidates - mean)
        
        # Adaptive Sigma (Simple decay)
        sigma *= 0.99
        
        current_best = np.max(fitness)
        if current_best > best_reward:
            best_reward = current_best
            best_params = candidates[np.argmax(fitness)]
        
        print(f"Generation {gen+1}: Best Average Reward = {best_reward:.2f}")

        # Early exit if we consistently hit very high scores
        if best_reward > 330:
            print("Target reached. Saving early.")
            break

    np.save(filename, best_params)
    print(f"Best policy saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--filename", type=str, default="best_policy.npy")
    args = parser.parse_args()

    if args.train:
        # population_size should be a multiple of your CPU core count for max efficiency
        train_and_save(args.filename, population_size=60, generations=150)