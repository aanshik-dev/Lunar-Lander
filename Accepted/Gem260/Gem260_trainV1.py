import gymnasium as gym
import numpy as np
import os

# Configuration for 300+ score
GENE_SIZE = 4996 
POP_SIZE = 150
GENERATIONS = 200
MUTATION_POWER = 0.05 

def policy_action(params, observation):
    # Same logic as policy_yourgroup.py to ensure consistency during training
    w1, b1 = params[0:512].reshape(8, 64), params[512:576]
    w2, b2 = params[576:4672].reshape(64, 64), params[4672:4736]
    w3, b3 = params[4736:4992].reshape(64, 4), params[4992:4996]
    
    a1 = np.maximum(0, np.dot(observation, w1) + b1)
    a2 = np.maximum(0, np.dot(a1, w2) + b2)
    return np.argmax(np.dot(a2, w3) + b3)

def evaluate(params):
    env = gym.make("LunarLander-v3")
    total_reward = 0
    episodes = 5 # Higher episodes during training reduces noise/luck
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_action(params, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    env.close()
    return total_reward / episodes

def train():
    # Initialize with small random weights
    current_params = np.random.randn(GENE_SIZE) * 0.1
    best_overall_score = -np.inf
    
    for gen in range(GENERATIONS):
        # Create noise/perturbations
        noise = np.random.randn(POP_SIZE, GENE_SIZE)
        rewards = []
        
        for i in range(POP_SIZE):
            candidate = current_params + noise[i] * MUTATION_POWER
            rewards.append(evaluate(candidate))
        
        rewards = np.array(rewards)
        best_idx = np.argmax(rewards)
        
        if rewards[best_idx] > best_overall_score:
            best_overall_score = rewards[best_idx]
            np.save("best_policy_Gm.npy", current_params + noise[best_idx] * MUTATION_POWER)

        # Standardization of rewards for better gradient estimation
        std = rewards.std()
        if std > 0:
            normalized_rewards = (rewards - rewards.mean()) / std
        else:
            normalized_rewards = rewards
            
        # Update weights using Evolution Strategy gradient
        current_params += 0.01 * np.dot(noise.T, normalized_rewards) / (POP_SIZE * MUTATION_POWER)
        
        print(f"Gen {gen}: Best Reward: {rewards[best_idx]:.2f} | Avg: {rewards.mean():.2f}")

if __name__ == "__main__":
    train()