#!/bin/bash

# Run the direct comparison script with default settings
echo "Running direct comparison with default settings..."
python direct_comparison.py --max-epochs 5 --num-samples 3

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Direct comparison completed successfully."
    echo "Results saved to direct_comparison_results/"
else
    echo "Direct comparison failed."
fi
