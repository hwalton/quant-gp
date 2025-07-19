from config import Config
import numpy as np

def _fit_preference_curve(w, points, poly_degree=3):
    """
    Helper function to fit straight lines through coordinate points
    
    Args:
        w: wealth value to evaluate
        points: list of (wealth, preference) tuples
        poly_degree: ignored (kept for compatibility)
    
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
    
    # Between points: simple linear interpolation
    return np.interp(w, wealth_points, pref_points)

# def get_preference_curve(cfg: Config):
#     if cfg.preference_curve == 'step_below_1000':
#         def step_below(w):
#             if w < 950:
#                 return 0
#             else:
#                 return 0.5
#         return step_below
    
#     elif cfg.preference_curve == 'step_above_1000':
#         def step_above(w):
#             if w < 1050:
#                 return 0
#             else:
#                 return 0.5
#         return step_above
    
#     elif cfg.preference_curve == 'not_below_920':
#         def not_below_920(w):
#             if w < 920:
#                 return 0
#             if 920 <= w < 4920:
#                 return 0.9 / 4000 * (w - 920)
#         return not_below_920

#     elif cfg.preference_curve == 'get_to_4500':
#         def get_to_4500(w):
#             if w < 900:
#                 return 0
#             if 900 <= w < 4900:
#                 return 0.9 / 4000 * (w - 900)
#             else:
#                 return 0.9
#         return get_to_4500
            
#     elif cfg.preference_curve == 'v_shape':
#         def v_shape(w):
#             if w < 1000:
#                 return 0.8
#             elif 1000 <= w < 2000:
#                 return 0.8 -0.8 / 1000 * (w-1000)
#             elif 1000 <= w < 2000:
#                 return 0
#             elif 2000 <= w < 3000:
#                 return 0.8 / 1000 * (w - 2000)
#             else:
#                 return 0.8
#         return v_shape
       
#     elif cfg.preference_curve == 'risk_averse':
#         gamma = cfg.gamma/1000
#         def risk_averse(w):
#             return np.tanh(((w/1000+1)**(1-gamma) - 1) / (1-gamma) if gamma != 1 else np.log(w))
#         return risk_averse

#     elif cfg.preference_curve == 'linear':
#         def linear(w):
#             return w
#         return linear

#     elif cfg.preference_curve == 'coordinate_points':
#         def coordinate_points(w):
#             # Get coordinate points from config
#             points = getattr(cfg, 'preference_points', [(600, -1), (1500,0.0), (5000, 0.5)])
#             poly_degree = getattr(cfg, 'preference_poly_degree', 3)
            
#             return _fit_preference_curve(w, points, poly_degree)
        
#         return coordinate_points

#     elif cfg.preference_curve == 'log_risk_averse':
#         def log_risk_averse(w):
#             # Scale-invariant logarithmic preference
#             # Maps log(w) to preference range [-0.9, 0.9]
            
#             # Define wealth range for mapping
#             w_min = getattr(cfg, 'log_w_min', 100)    # Minimum meaningful wealth
#             w_max = getattr(cfg, 'log_w_max', 10000)  # Maximum expected wealth
            
#             if w <= 0:
#                 return -0.9  # Very negative preference for zero/negative wealth
#             elif w <= w_min:
#                 return -0.9  # Very negative for wealth below minimum
#             elif w >= w_max:
#                 return 0.9   # Cap at maximum preference
#             else:
#                 # Logarithmic mapping: log(w) scaled to [-0.9, 0.9]
#                 log_w = np.log(w)
#                 log_min = np.log(w_min)
#                 log_max = np.log(w_max)
                
#                 # Normalize log(w) to [0, 1]
#                 normalized = (log_w - log_min) / (log_max - log_min)
                
#                 # Map to [-0.9, 0.9] range
#                 return -0.9 + 1.8 * normalized
        
#         return log_risk_averse
    
#     elif cfg.preference_curve == 'power_risk_averse':
#         def power_risk_averse(w):
#             # Power utility: w^(1-γ) where γ controls risk aversion
#             gamma = getattr(cfg, 'risk_aversion', 1)  # 1: risk tolerant, 5: risk averse
            
#             w_min = getattr(cfg, 'power_w_min', 10)
#             w_max = getattr(cfg, 'power_w_max', 10000)
            
#             if w <= 0:
#                 return -1
#             # elif w <= w_min:
#             #     return -1e5
#             # elif w >= w_max:
#             #     return 0.9
#             else:
#                 if gamma == 1.0:
#                     # Log utility case
#                     utility_val = np.log(w)
#                     utility_min = np.log(w_min)
#                     utility_max = np.log(w_max)
#                 else:
#                     # Power utility case
#                     utility_val = w**(1-gamma)
#                     utility_min = w_min**(1-gamma)
#                     utility_max = w_max**(1-gamma)
                
#                 # Normalize and map to [-0.9, 0.9]
#                 normalized = (utility_val - utility_min) / (utility_max - utility_min)
#                 return np.tanh(-0.9 + 1.8 * normalized)
        
#         return power_risk_averse


def get_preference_curve(cfg: Config):
    if cfg.preference_curve == 'coordinate_points':
        def coordinate_points(w):
            # Get coordinate points from config
            points = getattr(cfg, 'preference_points', [(6.39, -1), (7.31,0.0), (8.52, 0.5)])
            poly_degree = getattr(cfg, 'preference_poly_degree', 3)
        
            return _fit_preference_curve(w, points, poly_degree)
    
        return coordinate_points
    else:
        raise ValueError(f"Unsupported preference curve: {cfg.preference_curve}")
    


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
