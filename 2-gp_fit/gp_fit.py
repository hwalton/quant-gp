import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared

DATA_PATH = '../0-data/btc_monthly_prices.csv'
LOG_PKL_PATH = '../1-log-fit/log_trend_params.pkl'
GP_PKL_PATH = 'gp_model.pkl'
X_PRED_PKL = 'X_pred.npy'
Y_PRED_PKL = 'y_pred.npy'
YSTD_PKL = 'y_std.npy'
PLOT_PATH = 'gp_output.png'
MONTHS_INTO_FUTURE = 48
Y_LIMIT = (4, 18)

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    y = np.log(df['close'].astype(float).values)
    X = np.arange(len(y)).reshape(-1, 1)
    return X, y

def load_log_trend(pkl_path):
    params = joblib.load(pkl_path)
    def trend(x):
        return params[0] * np.log(params[1] * x + 1) + params[2]
    return trend

def build_kernel():
    return (
        C(1.0, (1e-3, 1e3)) *
        (
            RBF(length_scale=50.0, length_scale_bounds=(1e-1, 1e3)) +
            ExpSineSquared(length_scale=1.0, periodicity=48.0,
                           length_scale_bounds=(1e-2, 1e2),
                           periodicity_bounds=(12, 96))
        ) +
        WhiteKernel(noise_level=0.5, noise_level_bounds=(0.5, 10.0))
    )

def fit_gp(X, residuals, kernel):
    gp = GaussianProcessRegressor(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=3,
        normalize_y=True
    )
    gp.fit(X, residuals)
    return gp

def predict_gp(gp, X, months_ahead, trend_func):
    X_pred = np.linspace(0, len(X) + months_ahead, 700).reshape(-1, 1)
    y_resid_pred, y_std = gp.predict(X_pred, return_std=True)
    y_pred = y_resid_pred + trend_func(X_pred.ravel())
    return X_pred, y_pred, y_std

def save_outputs(gp, X_pred, y_pred, y_std):
    joblib.dump(gp, GP_PKL_PATH, compress=0)
    np.save(X_PRED_PKL, X_pred)
    np.save(Y_PRED_PKL, y_pred)
    np.save(YSTD_PKL, y_std)

def plot_results(X, y, X_pred, y_pred, y_std):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, y, 'kx', label='Observed BTC prices')
    ax.plot(X_pred, y_pred, 'b-', label='GP mean prediction')
    ax.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, label='1σ confidence')
    ax.set_xlabel('Weeks since start')
    ax.set_ylabel('Log(BTC Price (USD))')
    ax.set_title('Gaussian Process Regression on BTC Weekly Prices')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(*Y_LIMIT)
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close('all')

def main():
    X, y = load_data(DATA_PATH)
    log_trend = load_log_trend(LOG_PKL_PATH)
    residuals = y - log_trend(X.ravel())

    kernel = build_kernel()
    gp = fit_gp(X, residuals, kernel)
    X_pred, y_pred, y_std = predict_gp(gp, X, MONTHS_INTO_FUTURE, log_trend)

    save_outputs(gp, X_pred, y_pred, y_std)
    plot_results(X, y, X_pred, y_pred, y_std)

if __name__ == '__main__':
    main()
