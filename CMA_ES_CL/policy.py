# Filename: my_policy.py
# Neural network policy: 8 -> 64 -> 64 -> 4  (tanh, argmax)
# Parameters stored flat in best_policy.npy  (shape: 5508,)

import numpy as np

OBS_DIM = 8
ACT_DIM = 4
HIDDEN  = 64

def policy_action(params, observation):
    """
    params      : 1-D numpy array of shape (5508,)
    observation : 1-D numpy array of shape (8,)
    returns     : int in {0, 1, 2, 3}
    """
    idx = 0
    W1 = params[idx:idx + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN); idx += OBS_DIM * HIDDEN
    b1 = params[idx:idx + HIDDEN];                                      idx += HIDDEN
    W2 = params[idx:idx + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN);    idx += HIDDEN * HIDDEN
    b2 = params[idx:idx + HIDDEN];                                      idx += HIDDEN
    W3 = params[idx:idx + HIDDEN * ACT_DIM].reshape(HIDDEN, ACT_DIM);  idx += HIDDEN * ACT_DIM
    b3 = params[idx:idx + ACT_DIM]

    h1     = np.tanh(observation @ W1 + b1)
    h2     = np.tanh(h1         @ W2 + b2)
    logits = h2                 @ W3 + b3

    return int(np.argmax(logits))