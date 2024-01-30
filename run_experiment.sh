#!/bin/bash

echo "$HOSTNAME"
conda init bash &&
source activate pytorch 

# Set the number of times to run the Python script
# n=21
n=1

# Path to your Python script
python_script="./train_recognition.py"


# Loop to run the Python script n times
for ((i=0; i<$n; i++)); do
    echo "Running iteration $i"
    python "$python_script"  --model transformer --dataloader hamid --context $i
done

echo "Script completed $n iterations."