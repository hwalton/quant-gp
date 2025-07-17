from config import Config
import numpy as np

def get_utility_func(cfg: Config):
    if cfg.utility_function == 'identity' or cfg.utility_function == 'linear':
        return lambda w: w
    elif cfg.utility_function == 'log':
        return lambda w: np.log(w) if w > 0 else -np.inf
    elif cfg.utility_function == 'sqrt':
        return lambda w: np.sqrt(w) if w >= 0 else 0
    elif cfg.utility_function == 'step':
        return lambda w: 1.0 if w > cfg.step_threshold else 0.0
    elif cfg.utility_function == 'smooth_step':
        # Add numerical stability to prevent overflow
        def smooth_step(w):
            x = -cfg.step_steepness * (w - cfg.step_threshold)
            if x > 500:  # Prevent overflow
                return 0.0
            elif x < -500:
                return 1.0
            else:
                return 1 / (1 + np.exp(x))
        return smooth_step
    elif cfg.utility_function == 'sigmoid':
        return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
    elif cfg.utility_function == 'tanh':
        return lambda w: np.tanh((w - 800) / 20)
    elif cfg.utility_function == 'tanh_custom':
        return lambda w: np.tanh((w - 700) / 150) + 1 + w / 5000
    elif cfg.utility_function == 'crra':
        gamma = cfg.gamma
        return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
    else:
        raise ValueError(f"Unsupported utility function: {cfg.utility_function}")