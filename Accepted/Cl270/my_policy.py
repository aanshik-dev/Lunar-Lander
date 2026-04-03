import numpy as np

# Architecture: 8 -> 16 -> 4 (two-layer neural network)
# Parameter layout:
#   W1: 8x16  = 128 params  [0:128]
#   b1: 16    = 16  params  [128:144]
#   W2: 16x4  = 64  params  [144:208]
#   b2: 4     = 4   params  [208:212]
# Total: 212 parameters

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
    """Two-layer neural network policy.
    Compatible with evaluate_agent.py interface: policy_action(policy, observation) -> int
    """
    W1, b1, W2, b2 = _unpack(params)
    h = np.tanh(observation @ W1 + b1)
    logits = h @ W2 + b2
    return int(np.argmax(logits))