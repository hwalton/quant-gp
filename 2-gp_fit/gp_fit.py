import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared

@dataclass(frozen=True)
class Config:
    data_path: str ='../0-data/btc_weekly_prices.csv'
    log_pkl_path: str ='../1-log-fit/log_trend_params.pkl'
    gp_pkl_path: str ='gp_model.pkl'
    x_pred_pkl: str ='X_pred.npy'
    y_pred_pkl: str ='y_pred.npy'
    y_std_pkl: str ='y_std.npy'
    plot_path: str ='gp_output.png'
    points_into_future: int =48*7
    y_limit: tuple =(4, 18)

def load_data(cfg: Config):
    df = pd.read_csv(cfg.data_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    y = np.log(df['close'].astype(float).values)
    X = np.arange(len(y)).reshape(-1, 1)
    return X, y

def load_log_trend(cfg: Config):
    params = joblib.load(cfg.log_pkl_path)
    def trend(x):
        return params[0] * np.log(params[1] * x + 1) + params[2]
    return trend

def build_kernel():
    return (
            RBF(length_scale=200.0, length_scale_bounds=(10.0, 500.0)) +

            ExpSineSquared(length_scale=10.0, periodicity=208.0,
                           length_scale_bounds=(1.0, 100.0),
                           periodicity_bounds=(205, 210)) +
                           
        WhiteKernel(noise_level=0.25, noise_level_bounds=(0.1, 10.0))
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

def predict_gp(gp, X, trend_func, cfg: Config):
    X_pred = np.linspace(0, len(X) + cfg.points_into_future, 700).reshape(-1, 1)
    y_resid_pred, y_std = gp.predict(X_pred, return_std=True)
    y_pred = y_resid_pred + trend_func(X_pred.ravel())
    return X_pred, y_pred, y_std

def save_outputs(gp, X_pred, y_pred, y_std, cfg: Config):
    joblib.dump(gp, cfg.gp_pkl_path, compress=0)
    np.save(cfg.x_pred_pkl, X_pred)
    np.save(cfg.y_pred_pkl, y_pred)
    np.save(cfg.y_std_pkl, y_std)

def plot_results(X, y, X_pred, y_pred, y_std, cfg: Config):
    fig, ax = plt.subplots(figsize=(30, 18))
    ax.plot(X, y, 'kx', label='Observed BTC prices')
    ax.plot(X_pred, y_pred, 'b-', label='GP mean prediction')
    ax.fill_between(X_pred.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, label='1σ confidence')
    ax.set_xlabel('Weeks since start')
    ax.set_ylabel('Log(BTC Price (USD))')
    ax.set_title('Gaussian Process Regression on BTC Weekly Prices')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(*cfg.y_limit)
    fig.tight_layout()
    fig.savefig(cfg.plot_path)
    plt.close('all')

def main():
    cfg = Config()

    X, y = load_data(cfg)
    log_trend = load_log_trend(cfg)
    residuals = y - log_trend(X.ravel())

    kernel = build_kernel()
    gp = fit_gp(X, residuals, kernel)
    X_pred, y_pred, y_std = predict_gp(gp, X, log_trend, cfg)

    save_outputs(gp, X_pred, y_pred, y_std, cfg)
    plot_results(X, y, X_pred, y_pred, y_std, cfg)

if __name__ == '__main__':
    main()
