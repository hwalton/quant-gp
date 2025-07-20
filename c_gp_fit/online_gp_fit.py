import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import torch
import gpytorch
import time

from dataclasses import dataclass
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean  # Changed from LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from utils.utils import load_data

DATA_PATH = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')
LOG_PKL_PATH = os.path.join(PROJECT_ROOT, 'b_log_fit', 'log_trend_params.pkl')
GP_PKL_PATH = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'variational_gp_model.pth')
X_PRED_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'X_pred.npy')
Y_PRED_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'y_pred.npy')
Y_STD_PKL = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'y_std.npy')
PLOT_PATH = os.path.join(PROJECT_ROOT, 'c_gp_fit', 'variational_gp_output.png')

@dataclass(frozen=True)
class Config:
    data_path: str = DATA_PATH
    log_pkl_path: str = LOG_PKL_PATH
    gp_pkl_path: str = GP_PKL_PATH
    x_pred_pkl: str = X_PRED_PKL
    y_pred_pkl: str = Y_PRED_PKL
    y_std_pkl: str = 'y_std.npy'
    plot_path: str = PLOT_PATH
    points_into_future: int = 48*3
    y_limit: tuple =(4, 18)
    learning_rate: float = 0.1
    training_iter: int = 200  # Reduce from 500 to prevent parameter drift
    online_iter: int = 100  # or higher
    inducing_points: int = 150  # Increase from 100  # Number of inducing points

class VariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        
        # Use ConstantMean for zero-mean residuals
        self.mean_module = ConstantMean()
        
        # Use GPyTorch's stable kernels instead of custom implementation
        self.rbf_kernel = ScaleKernel(RBFKernel())
        
        # Try using PeriodicKernel - it's similar to ExpSineSquared but more stable
        self.periodic_kernel = ScaleKernel(PeriodicKernel())
        
        # Add kernels (sklearn uses +)
        self.covar_module = self.rbf_kernel + self.periodic_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class OnlineVariationalGP:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # WhiteKernel equivalent - separate noise
        self.likelihood = GaussianLikelihood()
        self.model = None
        self.train_x = None
        self.train_y = None
        self.optimizer = None
        self.mll = None
        
    def fit_initial(self, X, y):
        """Initial fit of the variational GP model"""
        print(f"Starting variational GP training with {len(y)} datapoints...")
        start_time = time.time()
        
        self.train_x = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
        self.train_y = torch.tensor(y, dtype=torch.float32)
        
        # Better inducing point placement - use quantiles
        inducing_points = torch.tensor(np.quantile(X, np.linspace(0, 1, self.cfg.inducing_points))).unsqueeze(-1).float()
        
        # Initialize model
        self.model = VariationalGPModel(inducing_points)
        
        print(f"Residual statistics (should be zero-mean):")
        print(f"  Mean: {y.mean():.6f}")
        print(f"  Std: {y.std():.4f}")  
        print(f"  Range: [{y.min():.4f}, {y.max():.4f}]")
        
        # MANUALLY SET KERNEL PARAMETERS - approximate sklearn
        print(f"\nSetting kernel parameters to approximate sklearn GP:")
        print(f"  RBF: outputscale={0.618**2:.3f}, lengthscale={3.8}")
        print(f"  Periodic: outputscale={0.641**2:.3f}, lengthscale={0.86}, period={187}")
        print(f"  Noise: {0.0142:.4f}")
        
        # RBF kernel - match sklearn exactly
        self.model.rbf_kernel.base_kernel.lengthscale = 3.8
        self.model.rbf_kernel.outputscale = 0.618**2
        
        # Periodic kernel - approximate ExpSineSquared
        self.model.periodic_kernel.base_kernel.lengthscale = 0.86
        self.model.periodic_kernel.base_kernel.period_length = 187.0
        self.model.periodic_kernel.outputscale = 0.641**2
        
        # Noise level - match sklearn exactly
        self.likelihood.noise = 0.0142
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)  # Higher learning rate for initial training
    
        self.mll = VariationalELBO(self.likelihood, self.model, num_data=len(y))
        
        # Train the model
        print(f"\nStarting variational training with {self.cfg.training_iter} iterations...")
        self._train_model_full()
        
        initial_training_time = time.time() - start_time
        print(f"✓ Initial variational training completed in {initial_training_time:.2f} seconds")
        print(f"  Training rate: {len(y)/initial_training_time:.1f} datapoints/second")
        print(f"  Using {self.cfg.inducing_points} inducing points")
        
        return self.model
    
    def add_datapoint(self, x_new, y_new):
        """Add a single new datapoint with efficient online update"""
        if self.model is None:
            raise ValueError("Model must be initially fitted before adding datapoints")
        
        print(f"Adding new datapoint ({x_new}, {y_new:.4f}) with online variational update...")
        start_time = time.time()
        
        # Convert new data to tensors
        x_new_tensor = torch.tensor([[x_new]], dtype=torch.float32)
        y_new_tensor = torch.tensor([y_new], dtype=torch.float32)
        
        # Add to training set
        self.train_x = torch.cat([self.train_x, x_new_tensor], dim=0)
        self.train_y = torch.cat([self.train_y, y_new_tensor], dim=0)
        
        # Update the MLL with new data size
        self.mll.num_data = len(self.train_y)
        
        # Efficient online update with fewer iterations
        self._train_model_online()
        
        update_time = time.time() - start_time
        print(f"✓ Datapoint added with variational update in {update_time:.2f} seconds")
        print(f"  Total training points: {len(self.train_y)}")
        print(f"  Update rate: {1/update_time:.1f} updates/second")
        
    def add_datapoint_sliding_window(self, x_new, y_new, window_size=500):
        """Add datapoint with sliding window for memory efficiency"""
        if self.model is None:
            raise ValueError("Model must be initially fitted before adding datapoints")
        
        print(f"Adding new datapoint ({x_new}, {y_new:.4f}) with sliding window update...")
        start_time = time.time()
        
        # Convert new data to tensors
        x_new_tensor = torch.tensor([[x_new]], dtype=torch.float32)
        y_new_tensor = torch.tensor([y_new], dtype=torch.float32)
        
        # Append new data
        self.train_x = torch.cat([self.train_x, x_new_tensor], dim=0)
        self.train_y = torch.cat([self.train_y, y_new_tensor], dim=0)
        
        # Keep only recent data for efficiency
        if len(self.train_y) > window_size:
            self.train_x = self.train_x[-window_size:]
            self.train_y = self.train_y[-window_size:]
        
        # Update the MLL with current data size
        self.mll.num_data = len(self.train_y)
        
        # Quick online update
        self._train_model_online()
        
        update_time = time.time() - start_time
        print(f"✓ Sliding window update completed in {update_time:.2f} seconds")
        print(f"  Window size: {len(self.train_y)} points")
        print(f"  Update rate: {1/update_time:.1f} updates/second")
    
    def _train_model_full(self):
        """Full training for initial fit"""
        self.model.train()
        self.likelihood.train()
        
        # # Optionally freeze kernel parameters to match sklearn exactly
        # for param in self.model.rbf_kernel.parameters():
        #     param.requires_grad = False
        # for param in self.model.periodic_kernel.parameters():
        #     param.requires_grad = False
        # self.likelihood.noise.requires_grad = False

        print(f"HW DEBUG: length of train_y: {len(self.train_y)}")
        for i in range(self.cfg.training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.optimizer.step()
            
            if (i + 1) % 25 == 0:
                noise_level = self.likelihood.noise.item()
                print(f'  Iter {i+1:3d}/{self.cfg.training_iter} - Loss: {loss.item():.3f} - Noise: {noise_level:.4f}')
    
    def _train_model_online(self):
        """Quick training for online updates"""
        self.model.train()
        self.likelihood.train()
        
        # Use fewer iterations for online updates
        for i in range(self.cfg.online_iter):
            self.optimizer.zero_grad()
            
            # For very large datasets, could use mini-batching here
            if len(self.train_y) > 1000:
                # Mini-batch approach for large datasets
                batch_size = 256
                indices = torch.randperm(len(self.train_y))[:batch_size]
                batch_x = self.train_x[indices]
                batch_y = self.train_y[indices]
                output = self.model(batch_x)
                loss = -self.mll(output, batch_y)
            else:
                # Use all data for smaller datasets
                output = self.model(self.train_x)
                loss = -self.mll(output, self.train_y)
            
            loss.backward()
            self.optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'    Online iter {i+1:2d}/{self.cfg.online_iter} - Loss: {loss.item():.3f}')
    
    def predict(self, X_pred):
        """Make predictions at new points"""
        print(f"Making variational predictions at {len(X_pred)} points...")
        start_time = time.time()
        
        self.model.eval()
        self.likelihood.eval()
        
        X_pred_tensor = torch.tensor(X_pred.reshape(-1, 1), dtype=torch.float32)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_pred_tensor))
            mean = predictions.mean.numpy()
            std = predictions.stddev.numpy()
        
        prediction_time = time.time() - start_time
        print(f"✓ Variational predictions completed in {prediction_time:.2f} seconds")
        print(f"  Prediction rate: {len(X_pred)/prediction_time:.1f} predictions/second")
            
        return mean, std
    
    def save_model(self, path):
        """Save the trained variational model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_x': self.train_x,
            'train_y': self.train_y,
            'inducing_points': self.model.variational_strategy.inducing_points
        }, path)
    
    def load_model(self, path):
        """Load a saved variational model"""
        print(f"Loading variational model from {path}...")
        start_time = time.time()
        
        checkpoint = torch.load(path)
        
        # Recreate model with saved inducing points
        inducing_points = checkpoint['inducing_points']
        self.model = VariationalGPModel(inducing_points)
        
        # Recreate training data
        self.train_x = checkpoint['train_x']
        self.train_y = checkpoint['train_y']
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        # Recreate optimizer and MLL
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.cfg.learning_rate)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.mll = VariationalELBO(self.likelihood, self.model, num_data=len(self.train_y))
        
        load_time = time.time() - start_time
        print(f"✓ Variational model loaded in {load_time:.3f} seconds")
        print(f"  Loaded {len(self.train_y)} training points")
        print(f"  Using {len(inducing_points)} inducing points")

def load_log_trend(cfg: Config):
    params = joblib.load(cfg.log_pkl_path)
    def trend(x):
        return params[0] * np.log(params[1] * x + 1) + params[2]
    return trend

def load_saved_predictions(cfg: Config):
    """Load saved GP predictions from files"""
    X_pred = np.load(cfg.x_pred_pkl)
    y_pred = np.load(cfg.y_pred_pkl) 
    y_std = np.load(cfg.y_std_pkl)
    return X_pred, y_pred, y_std

def fit_gp(X, residuals, cfg: Config):
    """Fit variational GP using GPyTorch - SAME APPROACH AS SKLEARN VERSION"""
    
    # DEBUG: Print what we're actually fitting (should be zero-mean residuals)
    print(f"\nDEBUG INFO:")
    print(f"X range: [{X.min():.0f}, {X.max():.0f}]")
    print(f"Residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")
    print(f"Residuals mean: {residuals.mean():.6f} (should be ~0)")
    print(f"Residuals std: {residuals.std():.4f}")
    
    gp = OnlineVariationalGP(cfg)
    gp.fit_initial(X, residuals)
    
    # Print optimized kernel parameters - compare to sklearn version
    print(f"\nOptimized variational kernel parameters:")
    print(f"RBF lengthscale: {gp.model.rbf_kernel.base_kernel.lengthscale.item():.3f}")
    print(f"RBF outputscale (σ²): {gp.model.rbf_kernel.outputscale.item():.3f}")
    print(f"Periodic lengthscale: {gp.model.periodic_kernel.base_kernel.lengthscale.item():.3f}")
    print(f"Periodic period: {gp.model.periodic_kernel.base_kernel.period_length.item():.3f}")
    print(f"Periodic outputscale (σ²): {gp.model.periodic_kernel.outputscale.item():.3f}")
    print(f"Noise level: {gp.likelihood.noise.item():.6f}")
    print("-" * 40)
    
    return gp

def predict_gp(gp, X, trend_func, cfg: Config):
    """SAME APPROACH: predict residuals, then add back trend"""
    X_pred = np.linspace(0, len(X) + cfg.points_into_future, 700)
    y_resid_pred, y_std = gp.predict(X_pred)
    y_pred = y_resid_pred + trend_func(X_pred)  # Add back the log trend
    return X_pred, y_pred, y_std

def save_outputs(gp, X_pred, y_pred, y_std, cfg: Config):
    gp.save_model(cfg.gp_pkl_path)
    np.save(cfg.x_pred_pkl, X_pred)
    np.save(cfg.y_pred_pkl, y_pred)
    np.save(cfg.y_std_pkl, y_std)

def timing_comparison_example(cfg: Config):
    """Compare timing of different update methods"""
    print("\n" + "="*60)
    print("VARIATIONAL GP TIMING COMPARISON")
    print("="*60)
    
    gp = OnlineVariationalGP(cfg)
    try:
        gp.load_model(cfg.gp_pkl_path)
        
        # Test standard online updates
        print("\n--- STANDARD ONLINE UPDATES ---")
        last_x = gp.train_x[-1].item()
        
        for i in range(3):
            new_x = last_x + i + 1
            new_y = np.random.normal(0, 0.1)
            gp.add_datapoint(new_x, new_y)
        
        # Test sliding window updates
        print("\n--- SLIDING WINDOW UPDATES ---")
        for i in range(3):
            new_x = last_x + i + 4
            new_y = np.random.normal(0, 0.1)
            gp.add_datapoint_sliding_window(new_x, new_y, window_size=300)
        
    except FileNotFoundError:
        print("No existing model found. Please run main() first to create initial model.")

def online_update_example(cfg: Config):
    """Example of variational online updates"""
    print("\n" + "="*50)
    print("VARIATIONAL ONLINE UPDATE EXAMPLE")
    print("="*50)
    
    gp = OnlineVariationalGP(cfg)
    try:
        gp.load_model(cfg.gp_pkl_path)
        
        # Add new datapoint with variational update
        last_x = gp.train_x[-1].item()
        new_x = last_x + 1
        new_y = 0.1
        
        gp.add_datapoint(new_x, new_y)
        
        # Make new predictions
        print("\nGenerating new variational predictions...")
        log_trend = load_log_trend(cfg)
        X_pred = np.linspace(0, new_x + cfg.points_into_future, 700)
        y_resid_pred, y_std = gp.predict(X_pred)
        y_pred = y_resid_pred + log_trend(X_pred)
        
        # Save updated results
        save_outputs(gp, X_pred, y_pred, y_std, cfg)
        print("Variational model updated and saved successfully!")
        
    except FileNotFoundError:
        print("No existing model found. Please run main() first to create initial model.")

def plot_results(cfg: Config, X, y):
    """Plot variational GP results"""
    
    # Load timestamps for axis labeling
    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    timestamps = df['timestamp'].values
    
    # Load saved predictions
    X_pred, y_pred, y_std = load_saved_predictions(cfg)
    
    # Only show the final half of the data for axis limits
    start_idx = len(X) - 364
    X_display = X[start_idx:]
    y_display = y[start_idx:]
    
    # But keep the full X_pred range for predictions
    X_pred_display = X_pred[X_pred >= X_display[0]]
    y_pred_display = y_pred[X_pred >= X_display[0]]
    y_std_display = y_std[X_pred >= X_display[0]]

    # Convert log prices back to actual prices for plotting
    y_actual = np.exp(y)
    y_display_actual = np.exp(y_display)
    y_pred_display_actual = np.exp(y_pred_display)
    y_std_upper_actual = np.exp(y_pred_display + y_std_display)
    y_std_lower_actual = np.exp(y_pred_display - y_std_display)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot observed data
    ax.plot(X, y_actual, 'kx', label='Historical BTC Prices', markersize=5, zorder=1)
    
    # Plot variational GP predictions
    ax.plot(X_pred_display, y_pred_display_actual, 'b-', label='Variational GP Mean', linewidth=2, zorder=2)
    ax.fill_between(X_pred_display, y_std_lower_actual, y_std_upper_actual, 
                    alpha=0.2, label='Variational GP 1σ Confidence Interval (68%)', color='skyblue', zorder=2)
    
    ax.set_yscale('log')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('BTC Price (USD)', fontsize=18)
    ax.set_title('Variational GP Regression on BTC Weekly Prices (Online Learning)', fontsize=21)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3, which='both')
    
    # Set axis limits
    ax.set_xlim(X_display[0], X_pred[-1])
    
    # Custom formatter
    def currency_formatter(y, p):
        if y >= 1000:
            return f'{int(y/1000)}k'
        else:
            return f'{int(y)}'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Calculate dynamic y-limits
    y_min = min(y_display_actual.min(), y_std_lower_actual.min())
    y_max = max(y_display_actual.max(), y_std_upper_actual.max())
    ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    fig.tight_layout()
    fig.savefig(cfg.plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"Variational GP plot saved to {cfg.plot_path}")

def plot_full_dataset_results(cfg: Config, X, y):
    """Plot GP results showing the FULL dataset in log space only, with extended log trend"""
    # Load timestamps for axis labeling
    df = pd.read_csv(cfg.data_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    timestamps = df['timestamp'].values
    
    # Load log trend function
    log_trend = load_log_trend(cfg)
    
    # Load saved predictions
    X_pred, y_pred, y_std = load_saved_predictions(cfg)
    
    # Flatten arrays if needed
    if X_pred.ndim > 1:
        X_pred = X_pred.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if y_std.ndim > 1:
        y_std = y_std.flatten()

    # Extend log trend to match prediction range
    X_trend = X_pred
    y_trend = log_trend(X_trend)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot observed data
    ax.plot(X, y, 'kx', label='Historical BTC Log Prices', markersize=3, alpha=0.7, zorder=1)
    # Plot extended log trend
    ax.plot(X_trend, y_trend, 'r-', label='Log Trend', linewidth=2, zorder=2)
    # Plot GP prediction
    ax.plot(X_pred, y_pred, 'b-', label='GP (Log Space)', linewidth=2, zorder=3)
    ax.fill_between(X_pred, y_pred - y_std, y_pred + y_std, 
                    alpha=0.2, label='GP 1σ (Log Space)', color='skyblue', zorder=2)
    
    ax.set_xlabel('Time (weeks)', fontsize=14)
    ax.set_ylabel('Log BTC Price', fontsize=14)
    ax.set_title('GP Fit: Full Dataset (Log Space)', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show years
    year_positions = []
    year_labels = []
    for i in range(0, len(timestamps), 52):  # Every ~52 weeks (1 year)
        if i < len(timestamps):
            year = pd.Timestamp(timestamps[i]).year
            year_positions.append(i)
            year_labels.append(str(year))
    
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save with different name
    full_plot_path = cfg.plot_path.replace('.png', '_full_dataset.png')
    fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"Full dataset log-space plot saved to {full_plot_path}")
    
    # Print some diagnostics
    print(f"\nDiagnostics:")
    print(f"  Data range: {X.min():.0f} to {X.max():.0f} weeks")
    print(f"  Log price range: {y.min():.2f} to {y.max():.2f}")
    print(f"  Residuals std: {(y - log_trend(X)).std():.4f}")
    print(f"  Prediction range: {X_pred.min():.0f} to {X_pred.max():.0f} weeks")

def main(cfg: Config = Config()):
    X, y = load_data(cfg)
    log_trend = load_log_trend(cfg)
    residuals = y - log_trend(X)  # SAME AS SKLEARN VERSION

    gp = fit_gp(X, residuals, cfg)
    X_pred, y_pred, y_std = predict_gp(gp, X, log_trend, cfg)

    save_outputs(gp, X_pred, y_pred, y_std, cfg)
    plot_results(cfg, X, y)           # Original zoomed plot
    plot_full_dataset_results(cfg, X, y)  # New full dataset plot

if __name__ == '__main__':    
    main()