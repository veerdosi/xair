#!/bin/bash
# Setup script for the XAI Chat Interface

# Exit on error
set -e

# Create necessary directories
mkdir -p output/api/cgrt
mkdir -p output/api/counterfactual
mkdir -p output/api/knowledge_graph
mkdir -p frontend/styles
mkdir -p frontend/js

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if Flask is installed
if ! pip list | grep -q "Flask"; then
    echo "Installing Flask and CORS for API server..."
    pip install Flask Flask-CORS
fi

# Make run.py executable
chmod +x run.py

# Set up application for development
export FLASK_APP=backend.api_router
export FLASK_DEBUG=1

echo ""
echo "Setup complete! You can now run the application with:"
echo "source venv/bin/activate"
echo "python run.py"
echo ""
echo "Options:"
echo "  --port PORT     Set the port (default: 5000)"
echo "  --no-browser    Don't open a browser window"
echo "  --debug         Run in debug mode"
echo "  --model MODEL   Specify the model name or path"
echo "  --skip-kg       Skip Knowledge Graph processing"
echo ""
