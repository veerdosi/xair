#!/bin/bash
# Run XAIR tests

# Set up virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install spaCy model if it's not already installed
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || python -m spacy download en_core_web_sm

# Run tests
echo "Running tests..."
pytest tests/ -v

# Check if any tests failed
if [ $? -ne 0 ]; then
  echo "Some tests failed."
  exit 1
fi

echo "All tests passed successfully!"
exit 0