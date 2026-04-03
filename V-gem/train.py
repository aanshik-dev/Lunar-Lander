import gymnasium as gym
import numpy as np
import multiprocessing as mp
from functools import partial

# Configuration for 320+ score
GENE_SIZE = 4996 
POP_SIZE = 128      # Increased for better gradient estimation
GENERATIONS = 1000 
INIT_LR = 0.05      # Higher starting LR
SIGMA = 0.03        # Lower sigma for finer precision
VERIFY_EPISODES = 20 # Stricter verification for 320+

def policy_action(params, observation):
    w1 = params[0:512].reshape(8, 64)
    b1 = params[512:576].reshape(64)
    w2 = params[576:4672].reshape(64, 64)
    b2 = params[4672:4736].reshape(64)
    w3 = params[4736:4992].reshape(64, 4)
    b3 = params[4992:4996].reshape(4)
    
    a1 = np.maximum(0, np.dot(observation, w1) + b1)
    a2 = np.maximum(0, np.dot(a1, w2) + b2)
    return np.argmax(np.dot(a2, w3) + b3)

def evaluate_single(params, episodes=5): 
    # Use 'render_mode=None' for speed during training
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
    lr = INIT_LR
    # Xavier/He initialization for better start
    current_params = np.random.randn(GENE_SIZE) * np.sqrt(2 / 8)
    best_overall_score = -np.inf
    
    # Initialize Multiprocessing Pool
    num_cpus = mp.cpu_count()
    print(f"Running with {num_cpus} workers...")
    pool = mp.Pool(processes=num_cpus)

    for gen in range(GENERATIONS):
        # Mirrored Sampling
        half_pop = POP_SIZE // 2
        epsilon_half = np.random.randn(half_pop, GENE_SIZE)
        epsilon = np.concatenate([epsilon_half, -epsilon_half], axis=0)
        
        candidates = [current_params + SIGMA * e for e in epsilon]
        
        # --- PARALLEL EVALUATION ---
        # Using map to distribute candidates across CPU cores
        rewards = pool.map(evaluate_single, candidates)
        rewards = np.array(rewards)
        # ---------------------------

        best_gen_idx = np.argmax(rewards)
        
        # High-precision verification if gen looks promising
        if rewards[best_gen_idx] > 280:
            v_score = evaluate_single(candidates[best_gen_idx], episodes=VERIFY_EPISODES)
            if v_score > best_overall_score:
                best_overall_score = v_score
                np.save("best_lander.npy", candidates[best_gen_idx])
                print(f"\n>>> NEW VERIFIED RECORD: {best_overall_score:.2f} <<<")

        # Update weights
        fitness = compute_ranks(rewards)
        gradient = np.dot(epsilon.T, fitness)
        current_params += lr * gradient / (POP_SIZE * SIGMA)
        
        # Adaptive LR Decay
        if gen % 20 == 0 and gen > 0:
            lr *= 0.98

        print(f"Gen {gen:03d} | Max: {rewards.max():.1f} | Mean: {rewards.mean():.1f} | Best: {best_overall_score:.1f} | LR: {lr:.4f}", end='\r')
        
        if best_overall_score > 320:
            print(f"\nTarget 320+ reached at Gen {gen}!")
            break
            
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Required for Windows/macOS multiprocessing
    train()