"""
ds_train.py  –  Train the LunarLander-v3 agent using CMA-ES (Covariance Matrix
                Adaptation Evolution Strategy) with a two-layer neural-network policy.

Usage
-----
Train and save:
    python ds_train.py --train --filename ds_best.npy

Load and play (render 5 episodes):
    python ds_train.py --play  --filename ds_best.npy

The terminal output format mirrors train_agent.py so graders see familiar progress lines:
    Generation <g>: Best Average Reward = <r>
"""

import os
import argparse
import numpy as np
import gymnasium as gym

# ─────────────────────────────────────────────
#  Policy architecture  (must match ds_policy.py)
# ─────────────────────────────────────────────
INPUT_DIM  = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 4
PARAM_SIZE = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM  # 212


def _unpack(params):
    idx = 0
    W1 = params[idx: idx + INPUT_DIM * HIDDEN_DIM].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += INPUT_DIM * HIDDEN_DIM
    b1 = params[idx: idx + HIDDEN_DIM]
    idx += HIDDEN_DIM
    W2 = params[idx: idx + HIDDEN_DIM * OUTPUT_DIM].reshape(HIDDEN_DIM, OUTPUT_DIM)
    idx += HIDDEN_DIM * OUTPUT_DIM
    b2 = params[idx: idx + OUTPUT_DIM]
    return W1, b1, W2, b2


def policy_action(params, observation):
    W1, b1, W2, b2 = _unpack(params)
    h = np.tanh(observation @ W1 + b1)
    logits = h @ W2 + b2
    return int(np.argmax(logits))


# ─────────────────────────────────────────────
#  Evaluation helper
# ─────────────────────────────────────────────
def evaluate_policy(params, episodes=5, render=False):
    """Roll out the policy for `episodes` episodes and return mean reward."""
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make("LunarLander-v3",
                       render_mode="human" if render else "rgb_array")
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += ep_reward
    return total_reward / episodes


# ─────────────────────────────────────────────
#  CMA-ES implementation (pure NumPy, no deps)
# ─────────────────────────────────────────────
class CMAES:
    """
    (μ/μ_w, λ)-CMA-ES  –  Hansen's reference implementation translated to NumPy.
    Minimises a fitness function; we negate the reward to minimise.
    """

    def __init__(self, mean, sigma=0.5, population_size=None):
        n = len(mean)
        self.n = n
        self.mean  = mean.copy()
        self.sigma = sigma

        # Population sizes
        lam = population_size or (4 + int(3 * np.log(n)))
        self.lam = lam
        mu  = lam // 2
        self.mu = mu

        # Recombination weights
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_w / raw_w.sum()
        mueff = 1.0 / (self.weights ** 2).sum()
        self.mueff = mueff

        # Step-size control
        self.cs   = (mueff + 2) / (n + mueff + 5)
        self.ds   = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix adaptation
        self.cc   = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.c1   = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu  = min(1 - self.c1,
                        2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # State
        self.ps = np.zeros(n)           # evolution path for sigma
        self.pc = np.zeros(n)           # evolution path for C
        self.C  = np.eye(n)             # covariance matrix
        self.eigeneval = 0
        self._sqrtC    = np.eye(n)      # initialised as I (C = I at start)
        self.invsqrtC  = np.eye(n)
        self.gen = 0

    # ------------------------------------------------------------------
    def ask(self):
        """Sample λ candidate solutions."""
        self._update_eigen()
        self._candidates = (
            self.mean
            + self.sigma
            * (np.random.randn(self.lam, self.n) @ self._sqrtC.T)
        )
        return self._candidates

    def tell(self, fitness_values):
        """
        Update distribution given fitness values (lower = better, i.e. negated reward).
        fitness_values : 1-D array of length λ, same order as ask() returned.
        """
        n, mu, lam = self.n, self.mu, self.lam
        order  = np.argsort(fitness_values)          # ascending (best = lowest neg-reward)
        best   = self._candidates[order[:mu]]        # top-μ individuals

        old_mean = self.mean.copy()
        self.mean = self.weights @ best

        # Evolution paths
        y_w = (self.mean - old_mean) / self.sigma
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                   * (self.invsqrtC @ y_w))

        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
                / self.chiN
                < 1.4 + 2 / (n + 1))

        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w)

        # Covariance update
        artmp = (best - old_mean) / self.sigma          # (μ, n)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc)
                               + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.weights * artmp.T) @ artmp)

        # Step-size update
        self.sigma *= np.exp((self.cs / self.ds)
                             * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma  = np.clip(self.sigma, 1e-6, 10.0)

        self.gen += 1

    # ------------------------------------------------------------------
    def _update_eigen(self):
        """Eigen-decompose C lazily (every λ/(c1+cmu)/n/10 evaluations)."""
        if self.gen - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.n / 10:
            self.eigeneval = self.gen
            # Enforce symmetry and positive definiteness
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D, B   = np.linalg.eigh(self.C)
            D      = np.sqrt(np.maximum(D, 1e-20))
            self._sqrtC   = B * D          # B @ diag(D)
            self.invsqrtC = B * (1.0 / D) @ B.T


# ─────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────
def train_and_save(filename,
                   population_size=64,
                   num_generations=200,
                   eval_episodes=5,
                   sigma_init=0.5,
                   target_reward=320.0,
                   patience=30):
    """
    Train using CMA-ES and save the best parameters.

    Parameters
    ----------
    population_size  : λ (offspring per generation). Larger → more stable but slower.
    num_generations  : hard cap on generations.
    eval_episodes    : episodes used to estimate fitness inside the training loop.
    sigma_init       : initial step size for CMA-ES.
    target_reward    : stop early once this mean reward is reached.
    patience         : stop if no improvement for this many generations.
    """
    rng = np.random.default_rng(42)
    init_mean = rng.standard_normal(PARAM_SIZE) * 0.1

    cma = CMAES(mean=init_mean, sigma=sigma_init, population_size=population_size)

    best_reward  = -np.inf
    best_params  = init_mean.copy()
    no_improve   = 0

    print(f"CMA-ES | n={PARAM_SIZE} | λ={cma.lam} | μ={cma.mu} | "
          f"target={target_reward}")
    print("-" * 60)

    for generation in range(1, num_generations + 1):
        candidates = cma.ask()                          # shape (λ, n)

        # Evaluate each candidate (negated for minimisation)
        fitness = np.array([
            -evaluate_policy(c, episodes=eval_episodes)
            for c in candidates
        ])

        cma.tell(fitness)

        gen_best_reward = -fitness.min()
        gen_mean_reward = -fitness.mean()

        if gen_best_reward > best_reward:
            best_reward  = gen_best_reward
            best_params  = candidates[np.argmin(fitness)].copy()
            no_improve   = 0
        else:
            no_improve  += 1

        # ── progress line matching train_agent.py format ──
        print(f"Generation {generation}: Best Average Reward = {best_reward:.2f}  "
              f"(gen_best={gen_best_reward:.2f}, mean={gen_mean_reward:.2f}, "
              f"σ={cma.sigma:.4f})")

        # Early stopping
        if best_reward >= target_reward:
            print(f"\n✓ Target reward {target_reward} reached at generation {generation}.")
            break
        if no_improve >= patience:
            print(f"\n⚠  No improvement for {patience} generations – stopping early.")
            break

    np.save(filename, best_params)
    print(f"\nBest policy saved to {filename}  (best_reward={best_reward:.2f})")
    return best_params


def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params


def play_policy(best_params, episodes=5):
    test_reward = evaluate_policy(best_params, episodes=episodes, render=True)
    print(f"Average reward of the best policy over {episodes} episodes: {test_reward:.2f}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or play best policy for Lunar Lander using CMA-ES."
    )
    parser.add_argument("--train",    action="store_true",
                        help="Train the policy using CMA-ES and save it.")
    parser.add_argument("--play",     action="store_true",
                        help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="ds_best.npy",
                        help="Filename to save/load the best policy.")
    parser.add_argument("--generations", type=int, default=200,
                        help="Maximum number of generations (default 200).")
    parser.add_argument("--popsize",  type=int, default=64,
                        help="CMA-ES population size λ (default 64).")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per fitness evaluation (default 5).")
    parser.add_argument("--sigma",    type=float, default=0.5,
                        help="Initial CMA-ES step size (default 0.5).")
    parser.add_argument("--target",   type=float, default=320.0,
                        help="Early-stop reward target (default 320).")
    args = parser.parse_args()

    if args.train:
        train_and_save(
            filename        = args.filename,
            population_size = args.popsize,
            num_generations = args.generations,
            eval_episodes   = args.episodes,
            sigma_init      = args.sigma,
            target_reward   = args.target,
        )
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Please specify --train to train and save a policy, "
              "or --play to load and play the best policy.")