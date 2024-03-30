#!/bin/bash

# Run nvidia-smi
echo "Running nvidia-smi..."
nvidia-smi

# Start Python and import NumPy
echo "Starting Python and importing NumPy..."
python -c "import numpy as np; print('NumPy imported successfully. Version:', np.__version__); import sys; sys.exit()" 

echo "Script completed."