import gymnasium as gym
import numpy as np
import os
from multiprocessing import Pool

# Configuration
GENE_SIZE = 4996 
POP_SIZE = 100    
GENERATIONS = 500 
LEARNING_RATE = 0.03
SIGMA = 0.05      
CHECKPOINT_FILE = "best.npy"

def policy_action(params, observation):
    # NN Architecture: 8 -> 64 -> 64 -> 4
    w1 = params[0:512].reshape(8, 64)
    b1 = params[512:576].reshape(64)
    w2 = params[576:4672].reshape(64, 64)
    b2 = params[4672:4736].reshape(64)
    w3 = params[4736:4992].reshape(64, 4)
    b3 = params[4992:4996].reshape(4)
    
    a1 = np.maximum(0, np.dot(observation, w1) + b1)
    a2 = np.maximum(0, np.dot(a1, w2) + b2)
    return np.argmax(np.dot(a2, w3) + b3)

def evaluate(params, episodes=8): 
    env = gym.make("LunarLander-v3")
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_action(params, obs)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            done = term or trunc
    env.close()
    return total_reward / episodes

def compute_ranks(x):
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    fitness = ranks.astype(float) / (len(x) - 1) - 0.5
    return fitness

def train():
    global LEARNING_RATE 
    
    # --- Checkpoint Loading ---
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading existing weights from {CHECKPOINT_FILE}...")
        current_params = np.load(CHECKPOINT_FILE)
        # Evaluate loaded params to set initial best_overall_score
        best_overall_score = evaluate(current_params, episodes=15)
        print(f"Initial Score from Checkpoint: {best_overall_score:.2f}")
    else:
        print("Starting training from scratch.")
        current_params = np.random.randn(GENE_SIZE) * np.sqrt(2 / 8)
        best_overall_score = -np.inf
    
    # Use Pool to handle parallel evaluations
    # It's best to use roughly the number of CPU cores you have (e.g., 4, 8, or 16)
    with Pool(processes=os.cpu_count()) as pool:
        for gen in range(GENERATIONS):
            half_pop = POP_SIZE // 2
            epsilon_half = np.random.randn(half_pop, GENE_SIZE)
            epsilon = np.concatenate([epsilon_half, -epsilon_half], axis=0)
            
            # Create candidates
            candidates = [current_params + SIGMA * e for e in epsilon]
            
            # --- Multiprocessing Step ---
            # pool.map distributes the candidates across available CPU cores
            rewards = pool.map(evaluate, candidates)
            rewards = np.array(rewards)
            
            best_gen_idx = np.argmax(rewards)
            
            # High-episode verification for potential best
            if rewards[best_gen_idx] > best_overall_score:
                verification_score = evaluate(candidates[best_gen_idx], episodes=15)
                if verification_score > best_overall_score:
                    best_overall_score = verification_score
                    np.save(CHECKPOINT_FILE, candidates[best_gen_idx])
                    print(f"--- New Verified Best: {best_overall_score:.2f} (Saved) ---")

            fitness = compute_ranks(rewards)
            gradient = np.dot(epsilon.T, fitness)
            current_params += LEARNING_RATE * gradient / (POP_SIZE * SIGMA)
            
            if gen % 50 == 0 and gen > 0:
                LEARNING_RATE *= 0.95

            print(f"Gen {gen}: Max {rewards.max():.2f} | Mean {rewards.mean():.2f} | LR {LEARNING_RATE:.4f}")
            
            if best_overall_score > 320:
                print("Target 320+ reached!")
                break

if __name__ == "__main__":
    train()