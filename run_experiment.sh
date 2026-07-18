#!/bin/bash

#SBATCH --job-name="MTRSAP All Modalities"
#SBATCH --error="./logs/job-%j-mtrsap_all_modalities.err"
#SBATCH --output="./logs/job-%j-mtrsap_all_modalities.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a40:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"
#SBATCH --chdir=/standard/UVA-DSA/Keshara/MTRSAP


TASK="${TASK:-Suturing}"
FIRST_MODALITY="${FIRST_MODALITY:-0}"
LAST_MODALITY="${LAST_MODALITY:-21}"

module purge
module load miniforge
source /home/cjh9fw/.bashrc
conda activate egoems

echo "Host: $HOSTNAME"
echo "Task: $TASK"
echo "Modalities: $FIRST_MODALITY-$LAST_MODALITY"

for ((modality=FIRST_MODALITY; modality<=LAST_MODALITY; modality++)); do
    echo "Starting modality $modality"

    python -u train_recognition.py \
        --model transformer \
        --dataloader v2 \
        --task "$TASK" \
        --modality "$modality"

    echo "Finished modality $modality"
done

echo "Completed modalities $FIRST_MODALITY-$LAST_MODALITY"
