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
python gp-model/run_pipeline.py --utility tanh_custom --wealth 1000 --offset 4

# Copy images to web directory
echo "2. Copying images to web directory..."
mkdir -p web/static/images
cp gp-model/outputs/*.png web/static/images/ 2>/dev/null || true

echo "Analysis complete! Images updated in web/static/images/"
echo "Generated files:"
ls -la web/static/images/*.png 2>/dev/null || echo "No PNG files found"
echo "Run 'make dev' to start the web server."
