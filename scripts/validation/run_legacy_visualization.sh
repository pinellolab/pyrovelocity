#!/bin/bash

# Run the legacy model visualization script with default settings
echo "Running legacy model visualization with default settings..."
source ../../.venv/bin/activate && python visualize_legacy_velocity.py --max-epochs 5 --num-samples 3

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Legacy model visualization completed successfully."
    echo "Results saved to legacy_visualization_results/"
else
    echo "Legacy model visualization failed."
fi
