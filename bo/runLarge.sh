#!/bin/bash

# List of available models
models=("meta-llama/Llama-2-70b-chat-hf" )

# Get list of JSONL files in the current directory
jsonl_files=(*.jsonl)

# Loop through all combinations of models and input files
for model in "${models[@]}"; do
    for input_file in "${jsonl_files[@]}"; do
        # Generate a unique filename for the output
        timestamp=$(date +%Y%m%d%H%M%S)
        base_input_file=$(basename "$input_file")
        output_file="output_${model//\//-}_$base_input_file_$timestamp.txt"

        echo "Running Python code with model: $model, input_file: $input_file, output will be saved in $output_file"
        
        # Execute the Python script in the background with nohup and redirect both stdout and stderr to the file
        nohup python inference.py --model "$model" --input_file_name "$input_file" > "$output_file" 2>&1 &

        # Capture the PID of the last background process (the Python script)
        pid=$!

        # Wait for the Python script to finish before proceeding to the next one
        wait $pid
    done
done