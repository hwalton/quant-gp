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
        def step_below(w):
            if w < 950:
                return 0.1
            else:
                return 0.9
        return step_below
    
    elif cfg.preference_curve == 'step_above_1000':
        def step_above(w):
            if w < 1050:
                return 0.1
            else:
                return 0.9
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
    
    elif cfg.preference_curve == 'not_below_800':
        def not_below_800(w):
            if w < 920:
                return -1
            if 920 <= w < 4920:
                return 1.9 / 4000 * (w - 920) - 1
            else:
                return 0.9
        return not_below_800


    
def get_utility_func(cfg: Config):
    def utility_func(w):
        preference_curve = get_preference_curve(cfg)
        numerically_stable_inf = 1e5
        if preference_curve(w) <= -1:
            return -numerically_stable_inf
        elif -1 < preference_curve(w) < 1:
            return np.arctanh(preference_curve(w))
        elif preference_curve(w) >= 1:
            return numerically_stable_inf
        else:
            raise ValueError(f"Invalid wealth value: {w}")
    return utility_func

# def get_utility_func(cfg: Config):
#     preference_curve = get_preference_curve(cfg)
#     return lambda w: (3*preference_curve(w))**5 + 3*preference_curve(w)