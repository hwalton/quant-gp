import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
X_PRED_PKL = '../2-gp_fit/X_pred.npy'
Y_PRED_PKL = '../2-gp_fit/y_pred.npy'
YSTD_PKL = '../2-gp_fit/y_std.npy'
ORIGINAL_LOG_CSV = '../0-data/btc_monthly_prices.csv'

INITIAL_WEALTH = 1.0
NORMALISE_RETURNS = True
UTILITY_FUNCTION = 'tanh'
SIGMOID_K = 25.0
W0 = 0.98
PREDICT_INDEX_OFFSET = 18
Y_LIMIT = (4, 18)

def load_data():
    X_pred = np.load(X_PRED_PKL)
    y_pred = np.load(Y_PRED_PKL)
    y_std = np.load(YSTD_PKL)

    df = pd.read_csv(ORIGINAL_LOG_CSV, sep=';')
    df = df.sort_values(by='timestamp')
    y_actual = np.log(df['close'].astype(float).values)

    return X_pred, y_pred, y_std, y_actual

def utility(w):
    if UTILITY_FUNCTION == 'log':
        return np.log(w) if w > 0 else -np.inf
    elif UTILITY_FUNCTION == 'sqrt':
        return np.sqrt(w) if w >= 0 else 0
    elif UTILITY_FUNCTION == 'sigmoid':
        return 1 / (1 + np.exp(-SIGMOID_K * (w - W0)))
    elif UTILITY_FUNCTION == 'tanh':
        return np.tanh(SIGMOID_K * (w - W0))
    else:
        raise ValueError(f"Unsupported utility function: {UTILITY_FUNCTION}")

def expected_utility(weight, mu, sigma):
    def integrand(r):
        wealth = INITIAL_WEALTH + weight * r
        return utility(wealth) * norm.pdf(r, loc=mu, scale=sigma)
    result, _ = quad(integrand, mu - 6 * sigma, mu + 6 * sigma, limit=100)
    return -result

def compute_gp_stats(X_pred, y_pred, y_std, y_actual):
    target_index = np.searchsorted(X_pred.ravel(), len(y_actual) + PREDICT_INDEX_OFFSET)
    price_now = y_actual[-1]
    price_pred = y_pred[target_index]
    mu = (price_pred - price_now) / price_now if NORMALISE_RETURNS else price_pred - price_now
    sigma = y_std[target_index] / price_now if NORMALISE_RETURNS else y_std[target_index]
    return mu, sigma

def optimise_allocation(mu, sigma):
    result = minimize_scalar(expected_utility, bounds=(0, 1), args=(mu, sigma), method='bounded')
    return result.x

def plot_expected_utility_curve(mu, sigma, optimal_weight):
    weights = np.linspace(0, 1, 100)
    utilities = [-expected_utility(w, mu, sigma) for w in weights]
    plt.figure(figsize=(8, 4))
    plt.plot(weights, utilities, label='Expected Utility')
    plt.axvline(optimal_weight, color='r', linestyle='--', label='Optimal weight')
    plt.xlabel('BTC Allocation')
    plt.ylabel('Expected Utility')
    plt.title('Expected Utility vs BTC Allocation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("utility_curve.png")

def plot_wealth_distribution(mu, sigma, optimal_weight):
    r_vals = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 1000)
    pdf_vals = norm.pdf(r_vals, loc=mu, scale=sigma)
    wealth_vals = INITIAL_WEALTH + optimal_weight * r_vals
    plt.figure(figsize=(8, 4))
    plt.plot(wealth_vals, pdf_vals, label='Wealth PDF')
    plt.axvline(INITIAL_WEALTH, color='r', linestyle='--', label='Initial Wealth')
    plt.xlabel('Simulated Future Wealth')
    plt.ylabel('Probability Density')
    plt.title('Wealth Distribution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("wealth_distribution.png")

def plot_utility_function():
    if UTILITY_FUNCTION == 'log':
        x_vals = np.linspace(0.01, 3, 500)
    elif UTILITY_FUNCTION == 'sqrt':
        x_vals = np.linspace(0, 3, 500)
    elif UTILITY_FUNCTION in ['sigmoid', 'tanh']:
        x_vals = np.linspace(W0 - 1.0, W0 + 1.0, 500)
    else:
        raise ValueError(f"Unsupported utility function: {UTILITY_FUNCTION}")

    y_vals = [utility(w) for w in x_vals]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label=f'{UTILITY_FUNCTION.capitalize()} utility', color='blue')

    if UTILITY_FUNCTION in ['sigmoid', 'tanh']:
        plt.axvline(W0, color='grey', linestyle='--', label='Inflection point (w0)')

    plt.xlabel('Wealth')
    plt.ylabel('Utility')
    plt.title('Utility Function')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("utility_func.png")

def main():
    X_pred, y_pred, y_std, y_actual = load_data()
    mu, sigma = compute_gp_stats(X_pred, y_pred, y_std, y_actual)

    print(f"\nExpected BTC return over {PREDICT_INDEX_OFFSET} months: {mu:.6f}")
    print(f"Predicted standard deviation: {sigma:.6f}")

    optimal_weight = optimise_allocation(mu, sigma)
    print(f"Optimal BTC allocation: {optimal_weight:.3f}")
    print(f"Optimal cash allocation: {1 - optimal_weight:.3f}")

    plot_expected_utility_curve(mu, sigma, optimal_weight)
    print("Saved utility_curve.png")

    plot_wealth_distribution(mu, sigma, optimal_weight)
    print("Saved wealth_distribution.png")

    plot_utility_function()
    print("Saved utility_func.png")

if __name__ == '__main__':
    main()
