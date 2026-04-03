import numpy as np

def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    b = params[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)