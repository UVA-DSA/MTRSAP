#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="MTRSAP - ICRA 2024"
#SBATCH --error="my_job_cheetah02.err"
#SBATCH --output="my_job_cheetah02.output"
#SBATCH --partition="gpu"
#SBATCH --nodelist="cheetah02"

echo "$HOSTNAME"
conda init zsh &&
source /u/cjh9fw/.bashrc &&
conda activate icra24 &&
python -u train_recognition.py --model transformer --dataloader v2 --modality 16 &&
echo "Done" &&
exit
