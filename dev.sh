#!/bin/bash

echo "Starting QuantGP development server..."

# Install Go dependencies
echo "Installing Go dependencies..."
go mod tidy

# Install Air if not already installed
if ! command -v air &> /dev/null; then
    echo "Installing Air for hot reloading..."
    go install github.com/cosmtrek/air@latest
fi

# Start the development server with hot reloading
echo "Starting development server on http://localhost:8080"
air
