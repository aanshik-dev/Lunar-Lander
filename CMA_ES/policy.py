# Filename: my_policy.py
import numpy as np

def policy_action(params, observation):
    """
    Neural Network Policy: 8 inputs -> 16 hidden (ReLU) -> 4 outputs.
    Total parameters: (8*16) + 16 + (16*4) + 4 = 128 + 16 + 64 + 4 = 212
    """
    # Layer 1: 8 -> 16
    W1 = params[0:128].reshape(8, 16)
    b1 = params[128:144].reshape(16)
    # Layer 2: 16 -> 4
    W2 = params[144:208].reshape(16, 4)
    b2 = params[208:212].reshape(4)
    
    # Forward Pass
    hidden = np.maximum(0, np.dot(observation, W1) + b1) # ReLU
    logits = np.dot(hidden, W2) + b2
    return np.argmax(logits)