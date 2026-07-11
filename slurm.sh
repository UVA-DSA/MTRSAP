#!/bin/bash

# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --job-name="EgoExoEMS MTRSAP Benchmark"
#SBATCH --error="./logs/job-%j-mtrsap_train_script.err"
#SBATCH --output="./logs/job-%j-mtrsap_train_script.output"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account="uva-dsa"

TASK="${TASK:-Suturing}"
MODALITY="${MODALITY:-12}"

# paths:
#   data_root: /standard/UVA-DSA/Robotic_Surgery_Datasets/MTRSAP
#   datasets_dir: Datasets
#   processed_datasets_dir: ProcessedDatasets
#   features_dir: Features
#   raw_dv_subdir: Datasets/dV
#   raw_jigsaws_subdir: Datasets/JIGSAWS
#   spatialcnn_subdir: Features/SpatialCNN
#   resnet_features_subdir: Features/resnet_features
#   image_features_subdir: Features/image_features
#   segmentation_features_subdir: Features/segmentation_masks/pca_features_normalized
#   segmentation_outputs_subdir: Features/segmentation_masks/outputs


module purge &&
module load miniforge  &&
source /home/cjh9fw/.bashrc  &&
echo "$HOSTNAME" &&
conda activate egoems &&


python -u train_recognition.py --model transformer --dataloader v2 --task "$TASK" --modality "$MODALITY" &&
echo "Done" &&
exit
