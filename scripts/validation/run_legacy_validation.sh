#!/bin/bash

# Run the legacy model validation script with default settings
echo "Running legacy model validation with default settings..."
source ../../.venv/bin/activate && python validate_legacy_model.py --max-epochs 5 --num-samples 3

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Legacy model validation completed successfully."
    echo "Results saved to legacy_validation_results/"
else
    echo "Legacy model validation failed."
fi
