#!/bin/bash

echo "Running QuantGP analysis pipeline..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -q -r requirements.txt

# Run analysis pipeline
echo "1. Running log trend fitting..."
cd 1-log-fit && python log-fit.py && cd ..

echo "2. Running Gaussian Process fitting..."
cd 2-gp_fit && python gp_fit.py && cd ..

echo "3. Running portfolio optimization..."
cd 3-optimise_portfolio && python optimise_portfolio.py && cd ..

# Copy images to web directory
echo "4. Copying images to web directory..."
cp 2-gp_fit/gp_output.png web/static/images/ 2>/dev/null || true
cp 3-optimise_portfolio/*.png web/static/images/ 2>/dev/null || true

echo "Analysis complete! Images updated in web/static/images/"
echo "Run 'make dev' to start the web server."
