# QuantGP - AI-Powered Portfolio Optimization

A modern web platform that combines advanced Gaussian Process modeling with utility theory to provide optimal Bitcoin portfolio allocation recommendations.

## Features

- **Gaussian Process Modeling**: Sophisticated probabilistic modeling that captures uncertainty and provides confidence intervals for predictions
- **Utility Theory Optimization**: Risk-aware portfolio optimization using utility functions that match your risk preferences
- **Real-time Analysis**: Monthly automated analysis with updated market data and portfolio recommendations
- **Modern Web Interface**: Clean, responsive design built with Go and Tailwind CSS

## Architecture

```
quant-gp/
├── cmd/                    # Application entry point
│   └── main.go
├── internal/               # Internal application code
│   └── handlers/           # HTTP handlers
├── web/                    # Web assets
│   ├── static/             # Static files (images, CSS, JS)
│   └── templates/          # HTML templates
├── gp-model/               # Modular analysis pipeline
│   ├── config.py           # Centralized configuration
│   ├── run_pipeline.py     # Main pipeline runner
│   ├── data/               # Data loading and migration
│   ├── log_fit/            # Log trend fitting
│   ├── gp_fit/             # Gaussian Process modeling
│   ├── portfolio/          # Portfolio optimization
│   └── outputs/            # Generated analysis outputs
├── 0-data/                 # Legacy data directory
├── 1-log-fit/              # Legacy log trend fitting
├── 2-gp_fit/               # Legacy GP modeling
└── 3-optimise_portfolio/   # Legacy portfolio optimization
```

## Quick Start

### Prerequisites

- Go 1.21+
- Python 3.8+ with required packages (see requirements.txt)
- Make (optional, for convenience commands)

### Development

1. **Clone and setup**:
   ```bash
   cd quant-gp
   make install    # Install Go dependencies and Air
   ```

2. **Run analysis** (generates charts):
   ```bash
   make analysis
   # Or use the new modular pipeline directly:
   python gp-model/run_pipeline.py --utility crra --wealth 1000 --offset 10 --gamma 2.0
   ```

3. **Start development server**:
   ```bash
   make dev        # Starts hot-reloading server on :8080
   ```

4. **Visit** http://localhost:8080

### Production Build

```bash
make build      # Creates bin/quantgp
make run        # Runs the built binary
```

## Development Workflow

1. **Update Analysis**: Run `make analysis` to regenerate charts with latest data
2. **Auto-reload**: The dev server automatically reloads on Go code changes
3. **View Results**: Charts are automatically displayed on the landing page

## API Endpoints

- `GET /` - Landing page
- `POST /signup` - Signup form submission
- `/static/` - Static assets (images, etc.)

## Technology Stack

**Backend**:
- Go with Gorilla Mux router
- Air for hot reloading during development

**Frontend**:
- Tailwind CSS for styling
- Alpine.js for interactive components
- Responsive design

**Analysis**:
- Python with NumPy, Pandas, Scikit-learn
- Gaussian Process Regression
- Utility theory optimization
- Matplotlib for visualization

## Configuration

The application can be configured through:
- `.air.toml` - Air hot reloading configuration  
- `Makefile` - Build and development commands
- `gp-model/config.py` - Centralized analysis configuration
- Command line arguments for the modular pipeline

### Analysis Configuration

The new modular pipeline supports flexible configuration:

```bash
python gp-model/run_pipeline.py \
  --utility crra \          # Utility function: log, crra, sqrt, etc.
  --wealth 1000 \           # Initial wealth amount
  --offset 10 \             # Prediction time offset
  --gamma 2.0               # Risk aversion parameter (for CRRA)
```

## Contributing

1. Make changes to Go code in `cmd/` or `internal/`
2. Update Python analysis in `gp-model/` directory (new modular structure)
3. Run `make analysis` or use the modular pipeline directly
4. Test with `make dev`

## Deployment

For production deployment:

1. Build the application: `make build`
2. Ensure Python analysis runs: `make analysis` 
3. Deploy the `bin/quantgp` binary with the `web/` directory
4. Set up automated analysis updates (cron job for `make analysis`)

The application serves static files and templates, so it can be deployed on any platform that supports Go binaries.
