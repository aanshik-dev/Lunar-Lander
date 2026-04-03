import numpy as np

def policy_action(params, observation):
    """
    Inference for 8-64-64-4 Neural Network.
    Compatible with evaluate_agent.py.
    """
    # Weight Extraction (Indices must match Training Script)
    w1 = params[0:512].reshape(8, 64)
    b1 = params[512:576].reshape(64)
    w2 = params[576:4672].reshape(64, 64)
    b2 = params[4672:4736].reshape(64)
    w3 = params[4736:4992].reshape(64, 4)
    b3 = params[4992:4996].reshape(4)

    # Forward Pass
    a1 = np.maximum(0, np.dot(observation, w1) + b1) # ReLU
    a2 = np.maximum(0, np.dot(a1, w2) + b2) # ReLU
    logits = np.dot(a2, w3) + b3
    
    return np.argmax(logits)