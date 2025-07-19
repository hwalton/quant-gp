from config import Config
import numpy as np

def _fit_preference_curve(w, points):
    # Sort points by wealth
    points = sorted(points, key=lambda x: x[0])
    
    if len(points) < 2:
        raise ValueError("Need at least 2 coordinate points")
    
    # Extract wealth and preference values
    wealth_points = np.array([p[0] for p in points])
    pref_points = np.array([p[1] for p in points])
    
    # Handle edge cases - vectorized
    w = np.asarray(w)  # Ensure w is a numpy array
    min_wealth = wealth_points[0]
    max_wealth = wealth_points[-1]
    min_pref = pref_points[0]
    max_pref = pref_points[-1]
    
    # Initialize result array
    result = np.zeros_like(w, dtype=float)
    
    # Before first point: horizontal at first preference value
    result[w <= min_wealth] = min_pref
    
    # After last point: horizontal at last preference value
    result[w >= max_wealth] = max_pref
    
    # Between points: linear interpolation in log-space for x-axis
    mask = (w > min_wealth) & (w < max_wealth)
    if np.any(mask):
        log_wealth_points = np.log(wealth_points)
        log_w = np.log(w[mask])
        result[mask] = np.interp(log_w, log_wealth_points, pref_points)
    
    return result

def get_preference_curve(cfg: Config):
    if cfg.preference_curve == 'step':
        def step_below(log_w):
            log_w = np.asarray(log_w)
            w = np.exp(log_w)
            return np.where(w < 900, -1, 0.9)
        return step_below

    elif cfg.preference_curve == 'coordinate_points':
        def coordinate_points(log_w):
            log_w = np.asarray(log_w)
            w = np.exp(log_w)  # Convert log back to wealth
            # Get coordinate points from config
            points = getattr(cfg, 'preference_points', [(100, -0.5), (10000, 0.8), (900, 0.5)])
            
            return _fit_preference_curve(w, points)
        
        return coordinate_points

    elif cfg.preference_curve == 'log_risk_averse':
        def log_risk_averse(log_w):
            log_w = np.asarray(log_w)
            w = np.exp(log_w)
            
            # Define wealth range for mapping
            w_min = getattr(cfg, 'log_w_min', 100)
            w_max = getattr(cfg, 'log_w_max', 1000)
            
            # Vectorized conditions
            result = np.full_like(w, -0.9, dtype=float)
            
            # Very negative preference for zero/negative wealth
            result[w <= 0] = -0.9
            
            # Very negative for wealth below minimum
            result[w <= w_min] = -0.9
            
            # Cap at maximum preference
            result[w >= w_max] = 0.9
            
            # Logarithmic mapping for values in between
            mask = (w > w_min) & (w < w_max)
            if np.any(mask):
                log_min = np.log(w_min)
                log_max = np.log(w_max)
                
                # Normalize log(w) to [0, 1]
                normalized = (log_w[mask] - log_min) / (log_max - log_min)
                
                # Map to [-0.9, 0.9] range
                result[mask] = -0.9 + 1.8 * normalized
            
            return result
        
        return log_risk_averse
    
    elif cfg.preference_curve == 'general_risk_level':
        def general_risk_level(log_w):
            log_w = np.asarray(log_w)
            w = np.exp(log_w)
            
            gamma = getattr(cfg, 'gamma', 5)
            w_min = getattr(cfg, 'power_w_min', 10)
            w_max = getattr(cfg, 'power_w_max', 10000)
            
            # Initialize result
            result = np.full_like(w, -1, dtype=float)
            
            # Handle positive wealth values
            mask = w > 0
            if np.any(mask):
                w_valid = w[mask]
                
                if gamma == 1.0:
                    # Log utility case
                    utility_val = np.log(w_valid)
                    utility_min = np.log(w_min)
                    utility_max = np.log(w_max)
                else:
                    # Power utility case
                    utility_val = w_valid**(1-gamma)
                    utility_min = w_min**(1-gamma)
                    utility_max = w_max**(1-gamma)
                
                # Normalize and map to [-0.9, 0.9]
                normalized = (utility_val - utility_min) / (utility_max - utility_min)
                result[mask] = np.tanh(-0.9 + 1.8 * normalized)
            
            return result
        
        return general_risk_level

def get_utility_func(cfg: Config):
    def utility_func(log_w):
        log_w = np.asarray(log_w)
        preference_curve = get_preference_curve(cfg)
        numerically_stable_inf = 1e5
        
        pref_values = preference_curve(log_w)
        
        # Vectorized utility calculation
        result = np.zeros_like(pref_values)
        
        # Handle different cases
        mask1 = pref_values <= -1
        mask2 = (pref_values > -1) & (pref_values < 1)
        mask3 = pref_values >= 1
        
        result[mask1] = -numerically_stable_inf
        result[mask2] = np.arctanh(pref_values[mask2])
        result[mask3] = numerically_stable_inf
        
        return result
    
    return utility_func
