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
echo "1. Running modular GP-based portfolio optimization..."
python gp-model/run_pipeline.py --utility crra --wealth 1000 --offset 10 --gamma 2.0

# Copy images to web directory
echo "2. Copying images to web directory..."
cp gp-model/outputs/*.png web/static/images/ 2>/dev/null || true

echo "Analysis complete! Images updated in web/static/images/"
echo "Run 'make dev' to start the web server."
