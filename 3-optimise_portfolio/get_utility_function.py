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
def _fit_preference_curve(w, points):
    """
    Helper function to fit straight lines through coordinate points on a semilogx scale
    
    Args:
        w: wealth value to evaluate
        points: list of (wealth, preference) tuples
    
    Returns:
        preference value at wealth w
    """
    # Sort points by wealth
    points = sorted(points, key=lambda x: x[0])
    
    if len(points) < 2:
        raise ValueError("Need at least 2 coordinate points")
    
    # Extract wealth and preference values
    wealth_points = np.array([p[0] for p in points])
    pref_points = np.array([p[1] for p in points])
    
    # Handle edge cases
    min_wealth = wealth_points[0]
    max_wealth = wealth_points[-1]
    min_pref = pref_points[0]
    max_pref = pref_points[-1]
    
    # Before first point: horizontal at first preference value
    if w <= min_wealth:
        return min_pref
    
    # After last point: horizontal at last preference value
    if w >= max_wealth:
        return max_pref
    
    # Between points: linear interpolation in log-space for x-axis
    # Convert wealth to log scale for interpolation
    log_wealth_points = np.log(wealth_points)
    log_w = np.log(w)
    
    # Linear interpolation in log-space
    return np.interp(log_w, log_wealth_points, pref_points)

def get_preference_curve(cfg: Config):
    if cfg.preference_curve == 'step':
        def step_below(log_w):
            w = np.exp(log_w)
            if w < 1150:
                return -1
            else:
                return 0.9
        return step_below

    elif cfg.preference_curve == 'coordinate_points':
        def coordinate_points(log_w):
            w= np.exp(log_w)  # Convert log back to wealth
            # Get coordinate points from config
            points = getattr(cfg, 'preference_points', [(100, -0.5), (10000, 0.1)])
            
            return _fit_preference_curve(w, points)
        
        return coordinate_points

    elif cfg.preference_curve == 'log_risk_averse':
        def log_risk_averse(log_w):
            w = np.exp(log_w)
            # Scale-invariant logarithmic preference
            # Maps log(w) to preference range [-0.9, 0.9]
            
            # Define wealth range for mapping
            w_min = getattr(cfg, 'log_w_min', 100)    # Minimum meaningful wealth
            w_max = getattr(cfg, 'log_w_max', 1000)  # Maximum expected wealth
            
            if w <= 0:
                return -0.9  # Very negative preference for zero/negative wealth
            elif w <= w_min:
                return -0.9  # Very negative for wealth below minimum
            elif w >= w_max:
                return 0.9   # Cap at maximum preference
            else:
                # Logarithmic mapping: log(w) scaled to [-0.9, 0.9]
                log_min = np.log(w_min)
                log_max = np.log(w_max)
                
                # Normalize log(w) to [0, 1]
                normalized = (log_w - log_min) / (log_max - log_min)
                
                # Map to [-0.9, 0.9] range
                return -0.9 + 1.8 * normalized
        
        return log_risk_averse
    
    elif cfg.preference_curve == 'general_risk_level':
        def general_risk_level(log_w):
            w = np.exp(log_w)
            # Power utility: w^(1-γ) where γ controls risk aversion
            gamma = getattr(cfg, 'gamma', 5)  # 1: risk tolerant, 5: risk averse
            
            w_min = getattr(cfg, 'power_w_min', 10)
            w_max = getattr(cfg, 'power_w_max', 10000)
            
            if w <= 0:
                return -1
            # elif w <= w_min:
            #     return -1e5
            # elif w >= w_max:
            #     return 0.9
            else:
                if gamma == 1.0:
                    # Log utility case
                    utility_val = np.log(w)
                    utility_min = np.log(w_min)
                    utility_max = np.log(w_max)
                else:
                    # Power utility case
                    utility_val = w**(1-gamma)
                    utility_min = w_min**(1-gamma)
                    utility_max = w_max**(1-gamma)
                
                # Normalize and map to [-0.9, 0.9]
                normalized = (utility_val - utility_min) / (utility_max - utility_min)
                return np.tanh(-0.9 + 1.8 * normalized)
        
        return general_risk_level
    
# def get_utility_func(cfg: Config):
#     def utility_func(w):
#         preference_curve = get_preference_curve(cfg)
#         numerically_stable_inf = 1e5
#         if preference_curve(w) == 0:
#             return -numerically_stable_inf
#         elif preference_curve(w) > 0:
#             return preference_curve(w) - 0.01 / preference_curve(w)
#         else:
#             raise ValueError(f"Invalid wealth value: {w}")
#     return utility_func

def get_utility_func(cfg: Config):
    def utility_func(w):
        preference_curve = get_preference_curve(cfg)
        numerically_stable_inf = 1e5
        if preference_curve(w) <= -1:
            return -numerically_stable_inf
        elif -1 < preference_curve(w) < 1:
            return np.arctanh(preference_curve(w)) 
            # return preference_curve(w)
        elif preference_curve(w) >= 1:
            return numerically_stable_inf
        else:
            raise ValueError(f"Invalid wealth value: {w}")
    return utility_func
