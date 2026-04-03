import numpy as np

def policy_action(params, observation):
    """
    Neural Network Policy: 8 inputs -> 64 hidden -> 64 hidden -> 4 outputs.
    Total parameters: (8*64 + 64) + (64*64 + 64) + (64*4 + 4) = 512 + 64 + 4096 + 64 + 256 + 4 = 5000.
    """
    # Layer 1: 8 -> 64
    w1 = params[0:512].reshape(8, 64)
    b1 = params[512:576].reshape(64)
    # Layer 2: 64 -> 64
    w2 = params[576:4672].reshape(64, 64)
    b2 = params[4672:4736].reshape(64)
    # Layer 3: 64 -> 4
    w3 = params[4736:4992].reshape(64, 4)
    b3 = params[4992:4996].reshape(4)

    # Forward Pass with ReLU activation
    z1 = np.dot(observation, w1) + b1
    a1 = np.maximum(0, z1) # ReLU
    
    z2 = np.dot(a1, w2) + b2
    a2 = np.maximum(0, z2) # ReLU
    
    logits = np.dot(a2, w3) + b3
    return np.argmax(logits)