#!/bin/bash

# Run the improved validation script with default settings
echo "Running improved validation with default settings..."
python run_improved_validation.py --max-epochs 5 --num-samples 3

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Improved validation completed successfully."
    echo "Results saved to improved_validation_results/"
else
    echo "Improved validation failed."
fi
