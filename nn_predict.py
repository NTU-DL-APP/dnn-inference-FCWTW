import numpy as np
import json

# === Activation functions ===
def relu(x):
    """
    Rectified Linear Unit activation function
    ReLU(x) = max(0, x)
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax activation function for multi-class classification
    Applies softmax along the last axis (classes)
    """
    # Subtract max for numerical stability (prevents overflow)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    
    # Compute exponentials
    exp_x = np.exp(x_shifted)
    
    # Normalize by sum of exponentials
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Dropout function ===
def dropout(x, rate=0.5, training=True):
    if not training or rate == 0:
        return x
    
    keep_prob = 1 - rate
    # Create binary mask and scale by keep_prob to maintain expected value
    mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
    return x * mask

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# === Updated inference function with dropout support ===
def nn_forward_h5(model_arch, weights, data, training=False):
    x = data
    for i, layer in enumerate(model_arch):
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
                # Apply dropout after ReLU activation (except for output layer)
                if training and 'dropout_rate' in cfg and i < len(model_arch) - 1:
                    x = dropout(x, cfg['dropout_rate'], training=True)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    