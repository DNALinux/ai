#!/bin/bash

# Validate input
validate_input() {
    if [[ ! $1 =~ ^[0-4](,[0-4])*$ ]]; then
        echo "Error: Invalid input. Please enter numbers between 0 and 4, separated by commas."
        exit 1
    fi
}

# Update the service file with CUDA_VISIBLE_DEVICES values
update_service() {
    # Check if CUDA_VISIBLE_DEVICES environment variable exists in the service file
    if grep -q '^Environment="CUDA_VISIBLE_DEVICES=' /etc/systemd/system/ollama.service; then
        # Update the existing CUDA_VISIBLE_DEVICES values
        sudo sed -i 's/^Environment="CUDA_VISIBLE_DEVICES=.*/Environment="CUDA_VISIBLE_DEVICES='"$1"'"/' /etc/systemd/system/ollama.service
    else
        # Add a new CUDA_VISIBLE_DEVICES environment variable
        sudo sed -i '/\[Service\]/a Environment="CUDA_VISIBLE_DEVICES='"$1"'"' /etc/systemd/system/ollama.service
    fi

    # Reload and restart the systemd service
    sudo systemctl daemon-reload
    sudo systemctl restart ollama.service

    echo "Service updated and restarted with CUDA_VISIBLE_DEVICES=$1"
}

# Check if arguments are passed
if [ "$#" -eq 0 ]; then
    # Prompt user for CUDA_VISIBLE_DEVICES values if no arguments are passed
    read -p "Enter CUDA_VISIBLE_DEVICES values (0-4, comma-separated): " cuda_values
    validate_input "$cuda_values"
    update_service "$cuda_values"
else
    # Use arguments as CUDA_VISIBLE_DEVICES values
    cuda_values="$1"
    validate_input "$cuda_values"
    update_service "$cuda_values"
fi
