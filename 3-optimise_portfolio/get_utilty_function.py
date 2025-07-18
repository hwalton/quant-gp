from config import Config
import numpy as np

# def get_utility_func(cfg: Config):
#     if cfg.utility_function == 'identity' or cfg.utility_function == 'linear':
#         return lambda w: w
#     elif cfg.utility_function == 'log':
#         return lambda w: np.log(w) if w > 0 else -np.inf
#     elif cfg.utility_function == 'sqrt':
#         return lambda w: np.sqrt(w) if w >= 0 else 0
#     elif cfg.utility_function == 'step':
#         return lambda w: 1.0 if w > cfg.step_threshold else 0.0
#     elif cfg.utility_function == 'smooth_step':
#         # Add numerical stability to prevent overflow
#         def smooth_step(w):
#             x = -cfg.step_steepness * (w - cfg.step_threshold)
#             if x > 500:  # Prevent overflow
#                 return 0.0
#             elif x < -500:
#                 return 1.0
#             else:
#                 return 1 / (1 + np.exp(x))
#         return smooth_step
#     elif cfg.utility_function == 'sigmoid':
#         return lambda w: 1 / (1 + np.exp(-cfg.sigmoid_k * (w - cfg.w0)))
#     elif cfg.utility_function == 'tanh':
#         return lambda w: np.tanh((w - 800) / 20)
#     elif cfg.utility_function == 'tanh_custom':
#         return lambda w: np.tanh((w - 700) / 150) + 1 + w / 5000
#     elif cfg.utility_function == 'crra':
#         gamma = cfg.gamma
#         return lambda w: (w**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w)
#     else:
#         raise ValueError(f"Unsupported utility function: {cfg.utility_function}")


def get_preference_curve(cfg: Config):
    if cfg.preference_curve == 'step_below_1000':
        # Sharp drop below 1000, constant above
        def step_below(w):
            if w < 1000:
                return 0.1
            else:
                return 0.9
        return step_below
    
    elif cfg.preference_curve == 'step_above_1000':
        # Constant below 1000, sharp rise above
        def step_above(w):
            if w < 1000:
                return 0.3
            else:
                return 0.95
        return step_above
    
    elif cfg.preference_curve == 'smooth_step':
        # Smooth S-curve transition around 1000
        def smooth_step(w):
            steepness = 0.01  # Controls transition sharpness
            center = 1000
            base = 0.2
            height = 0.7
            return base + height / (1 + np.exp(-steepness * (w - center)))
        return smooth_step
    
    elif cfg.preference_curve == 'fast_climb_drop':
        # Fast climb from 0 to 2000, then fast drop but never touching 0
        def fast_climb_drop(w):
            if w <= 0:
                return 0.05
            elif w <= 2000:
                # Fast rise to peak at 2000
                return 0.05 + 0.9 * (1 - np.exp(-w / 500))
            else:
                # Fast exponential decay but bounded above 0.1
                decay = 0.95 * np.exp(-(w - 2000) / 800)
                return max(0.1, decay)
        return fast_climb_drop
    
    elif cfg.preference_curve == 'risk_averse':
        # Steep rise up to 1500, then very slow growth (risk averse)
        def risk_averse(w):
            if w <= 0:
                return 0.01
            elif w <= 1500:
                # Steep concave rise
                return 0.01 + 0.85 * (1 - np.exp(-w / 400))
            else:
                # Very slow logarithmic growth
                return 0.86 + 0.13 * np.log(w / 1500) / 10
        return np.vectorize(risk_averse)
    
    elif cfg.preference_curve == 'loss_averse':
        # Very sensitive to losses below 1000, moderate gains above
        def loss_averse(w):
            if w <= 0:
                return 0.001
            elif w < 1000:
                # Steep punishment for losses
                loss_ratio = w / 1000
                return 0.001 + 0.499 * loss_ratio**3  # Cubic penalty
            else:
                # Moderate square root growth for gains
                gain_ratio = (w - 1000) / 4000  # Normalize to [0,1] for w in [1000, 5000]
                return 0.5 + 0.49 * np.sqrt(min(1.0, gain_ratio))
        return loss_averse
    
    elif cfg.preference_curve == 'target_seeking':
        # Peaks at target wealth (2500), drops on both sides
        def target_seeking(w):
            if w <= 0:
                return 0.1
            else:
                target = 2500
                spread = 1500
                # Gaussian-like curve centered at target
                distance = abs(w - target)
                peak_value = 0.95
                base_value = 0.15
                return base_value + (peak_value - base_value) * np.exp(-(distance / spread)**2)
        return target_seeking
    
    else:
        # Default: simple sigmoid
        def default_sigmoid(w):
            return 0.1 + 0.8 / (1 + np.exp(-0.005 * (w - 1000)))
        return default_sigmoid

    
def get_utility_func(cfg: Config):
    preference_curve = get_preference_curve(cfg)
    return lambda w: np.arctanh(preference_curve(w)*0.99999999999)