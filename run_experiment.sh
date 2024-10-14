#!/bin/bash

echo "$HOSTNAME"
conda init bash &&
source activate pytorch 

# n is the total number of different modality combinations (0-21)
# n=21

# Path to your Python script
python_script="./train_recognition.py"

# Loop to run the Python script n times
for ((i=0; i<=$n; i++)); do
    echo "Running iteration $i"
    python "$python_script"  --model transformer  --dataloader v2 --modality $i &&
done

echo "Script completed $n iterations."