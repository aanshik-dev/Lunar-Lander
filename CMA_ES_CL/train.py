# Filename: train_agent.py
# Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# + Neural Network policy (8 -> 64 -> 64 -> 4)
# + Parallel fitness evaluation using multiprocessing
# CMA-ES is far more reliable than vanilla GA or naive PPO for this task.

import gymnasium as gym
import numpy as np
import argparse
import os
from multiprocessing import Pool, cpu_count, freeze_support

# ─────────────────────────────────────────────────────────────────
#  NETWORK LAYOUT  (must match my_policy.py exactly)
#  8 -> 64 -> 64 -> 4
#  Param count: 8*64 + 64 + 64*64 + 64 + 64*4 + 4 = 5508
# ─────────────────────────────────────────────────────────────────
OBS_DIM    = 8
ACT_DIM    = 4
HIDDEN     = 64
PARAM_SIZE = (OBS_DIM * HIDDEN + HIDDEN +
               HIDDEN * HIDDEN + HIDDEN +
               HIDDEN * ACT_DIM + ACT_DIM)

def unpack_params(params):
    idx = 0
    W1 = params[idx:idx + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN); idx += OBS_DIM * HIDDEN
    b1 = params[idx:idx + HIDDEN];                                      idx += HIDDEN
    W2 = params[idx:idx + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN);    idx += HIDDEN * HIDDEN
    b2 = params[idx:idx + HIDDEN];                                      idx += HIDDEN
    W3 = params[idx:idx + HIDDEN * ACT_DIM].reshape(HIDDEN, ACT_DIM);  idx += HIDDEN * ACT_DIM
    b3 = params[idx:idx + ACT_DIM]
    return W1, b1, W2, b2, W3, b3

def policy_action(params, observation):
    """Deterministic greedy action — used by evaluate_agent.py."""
    W1, b1, W2, b2, W3, b3 = unpack_params(params)
    h1     = np.tanh(observation @ W1 + b1)
    h2     = np.tanh(h1         @ W2 + b2)
    logits = h2                 @ W3 + b3
    return int(np.argmax(logits))

# ─────────────────────────────────────────────────────────────────
#  FITNESS EVALUATION  (worker function — must be top-level for pickling)
# ─────────────────────────────────────────────────────────────────
def _eval_worker(args):
    """Evaluate one parameter vector over n_episodes. Returns mean reward."""
    params, n_episodes, seed = args
    rng   = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_episodes):
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done   = False
        ep_rew = 0.0
        while not done:
            action = policy_action(params, obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_rew += rew
            done = terminated or truncated
        env.close()
        total += ep_rew
    return total / n_episodes

def _eval_population(population, n_episodes, pool):
    seeds = np.random.randint(0, 2**31, size=len(population))
    args  = [(ind, n_episodes, int(s)) for ind, s in zip(population, seeds)]
    return np.array(pool.map(_eval_worker, args))

# ─────────────────────────────────────────────────────────────────
#  CMA-ES  (pure numpy)
# ─────────────────────────────────────────────────────────────────
class CMAES:
    """(μ/μ_w, λ)-CMA-ES. Maximises fitness."""
    def __init__(self, dim, popsize=None, sigma0=0.5, mu=None):
        self.dim  = dim
        self.lam  = popsize if popsize else 4 + int(3 * np.log(dim))
        self.mu   = mu      if mu      else self.lam // 2

        raw_w      = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w     = raw_w / raw_w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()

        self.sigma = sigma0
        self.cs    = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds    = 1 + 2*max(0, np.sqrt((self.mueff-1)/(dim+1))-1) + self.cs
        self.chiN  = np.sqrt(dim)*(1 - 1/(4*dim) + 1/(21*dim**2))

        self.cc    = (4 + self.mueff/dim) / (dim + 4 + 2*self.mueff/dim)
        self.c1    = 2 / ((dim+1.3)**2 + self.mueff)
        self.cmu   = min(1-self.c1,
                         2*(self.mueff-2+1/self.mueff)/((dim+2)**2+self.mueff))

        self.mean       = np.zeros(dim)
        self.ps         = np.zeros(dim)
        self.pc         = np.zeros(dim)
        self.C          = np.eye(dim)
        self.D          = np.ones(dim)
        self.B          = np.eye(dim)
        self.invsqrtC   = np.eye(dim)
        self.eigeneval  = 0
        self.counteval  = 0

    def ask(self):
        self._update_eigensystem()
        zs = np.random.randn(self.lam, self.dim)
        ys = zs @ (self.B * self.D).T
        xs = self.mean + self.sigma * ys
        return xs, ys

    def tell(self, xs, ys, fitnesses):
        self.counteval += self.lam
        order    = np.argsort(fitnesses)[::-1]
        xbest    = xs[order[:self.mu]]
        ybest    = ys[order[:self.mu]]
        old_mean = self.mean.copy()
        self.mean = self.w @ xbest

        self.ps = ((1-self.cs)*self.ps
                   + np.sqrt(self.cs*(2-self.cs)*self.mueff)
                   * self.invsqrtC @ (self.mean-old_mean)/self.sigma)

        hs = (np.linalg.norm(self.ps)
              / np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lam))
              / self.chiN) < 1.4 + 2/(self.dim+1)

        self.pc = ((1-self.cc)*self.pc
                   + hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)
                   * (self.mean-old_mean)/self.sigma)

        artmp  = ybest
        self.C = ((1-self.c1-self.cmu)*self.C
                  + self.c1*(np.outer(self.pc,self.pc)
                             + (1-hs)*self.cc*(2-self.cc)*self.C)
                  + self.cmu*(artmp.T*self.w)@artmp)

        self.sigma *= np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps)/self.chiN-1))

    def _update_eigensystem(self):
        thresh = self.lam/(self.c1+self.cmu)/self.dim/10
        if self.counteval - self.eigeneval > thresh:
            self.eigeneval = self.counteval
            self.C  = np.triu(self.C) + np.triu(self.C,1).T
            D2, self.B = np.linalg.eigh(self.C)
            self.D  = np.sqrt(np.maximum(D2, 1e-20))
            self.invsqrtC = (self.B/self.D) @ self.B.T

# ─────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────
def train(filename,
          n_generations = 200,
          sigma0        = 0.5,
          popsize       = None,
          episodes_fast = 5,
          episodes_eval = 20,
          n_workers     = None):

    n_workers = n_workers or min(cpu_count(), 16)
    dim       = PARAM_SIZE

    if popsize is None:
        popsize = max(32, 4 + int(3 * np.log(dim)))

    cma       = CMAES(dim=dim, popsize=popsize, sigma0=sigma0)
    cma.mean  = np.random.randn(dim) * 0.1

    best_params = cma.mean.copy()
    best_reward = -np.inf

    print(f"CMA-ES  |  dim={dim}  popsize={popsize}  workers={n_workers}  sigma0={sigma0}")
    print(f"{'Gen':>5}  {'BestInGen':>10}  {'MeanGen':>10}  {'BestEver':>10}  {'Sigma':>10}")
    print("-" * 56)

    with Pool(processes=n_workers) as pool:
        for gen in range(1, n_generations + 1):
            xs, ys = cma.ask()
            fits   = _eval_population(xs, episodes_fast, pool)
            cma.tell(xs, ys, fits)

            best_idx = int(np.argmax(fits))

            # More careful eval of the generation's best candidate
            eval_rew = _eval_worker((xs[best_idx], episodes_eval,
                                     int(np.random.randint(0, 2**31))))
            if eval_rew > best_reward:
                best_reward = eval_rew
                best_params = xs[best_idx].copy()
                np.save(filename, best_params)

            print(f"{gen:>5}  {fits[best_idx]:>10.2f}  {fits.mean():>10.2f}"
                  f"  {best_reward:>10.2f}  {cma.sigma:>10.4f}")

            # Adaptive: use more episodes once we're in a promising region
            if best_reward >= 200 and episodes_fast < 10:
                episodes_fast = 10
                print(f"  [Switched to {episodes_fast} episodes/eval]")
            if best_reward >= 280 and episodes_fast < 15:
                episodes_fast = 15
                print(f"  [Switched to {episodes_fast} episodes/eval]")

            if best_reward >= 320:
                print(f"\n  Target reached at generation {gen}. Stopping early.")
                break

    print("-" * 56)
    print(f"Training complete.  Best reward: {best_reward:.2f}")
    print(f"Best policy saved to {filename}")
    return best_params

# ─────────────────────────────────────────────────────────────────
#  PLAY (rendered)
# ─────────────────────────────────────────────────────────────────
def play(params, episodes=5):
    total = 0.0
    for ep in range(episodes):
        env = gym.make("LunarLander-v3", render_mode="human")
        obs, _ = env.reset()
        done   = False
        ep_rew = 0.0
        while not done:
            action = policy_action(params, obs)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_rew += rew
            done = terminated or truncated
        env.close()
        total += ep_rew
        print(f"  Episode {ep+1}: {ep_rew:.2f}")
    print(f"Average reward over {episodes} episodes: {total/episodes:.2f}")

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(
        description="Train or play Lunar Lander policy using CMA-ES.")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--play",        action="store_true")
    parser.add_argument("--filename",    type=str,   default="best_policy.npy")
    parser.add_argument("--generations", type=int,   default=200)
    parser.add_argument("--sigma0",      type=float, default=0.5)
    parser.add_argument("--popsize",     type=int,   default=None,
                        help="Population size (default: auto ~32+)")
    parser.add_argument("--workers",     type=int,   default=None,
                        help="Parallel workers (default: cpu_count)")
    args = parser.parse_args()

    if args.train:
        train(
            filename      = args.filename,
            n_generations = args.generations,
            sigma0        = args.sigma0,
            popsize       = args.popsize,
            n_workers     = args.workers,
        )
    elif args.play:
        if not os.path.exists(args.filename):
            print(f"File '{args.filename}' not found.")
        else:
            params = np.load(args.filename)
            print(f"Loaded policy from '{args.filename}'  (shape={params.shape})")
            play(params, episodes=5)
    else:
        print("Specify --train to train, or --play to watch the agent.")