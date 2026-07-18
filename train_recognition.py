import torch
import numpy as np
import json
from data import get_dataloaders, generate_data
from data import kinematic_feature_names, trajectory_feature_names, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables
from config import modality_mapping, learning_params, dataloader_params, transformer_params, tcn_model_params, RECORD_RESULTS, data_paths
from models.utils import reset_parameters, traintest_loop, rolling_average
from models import initiate_model
from utils import json_to_csv

import datetime
import argparse
import os
from copy import deepcopy


torch.manual_seed(0)


# end of imports #


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A simple command-line argument parser")

# Add arguments
parser.add_argument("--model", help="Specify which model to run", required=True)
parser.add_argument("--dataloader", help="Specify which dataloader", required=True)
parser.add_argument("--modality", help="Specify which modality combo", required=True, type=int)
parser.add_argument("--task", help="Specify the surgical task", default="Suturing")
# parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
model_name = args.model
dataloader = args.dataloader
context = args.modality
task = args.task
# verbose_mode = args.verbose

run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
job_id = os.environ.get("SLURM_JOB_ID", "local")
run_name = f"{run_timestamp}_{model_name}_job-{job_id}"
run_results_dir = os.path.join(
    "results", "recognition", task, f"modality_{context}", run_name
)


# manual seeding ensure reproducibility
# torch.manual_seed(0)



# context = dataloader_params["context"]

if context in modality_mapping:
    Features, include_resnet_features, include_colin_features, include_segmentation_features = modality_mapping[context]
else:
    print("Invalid modality choice!")
    exit(-1)

# Recognition does not consume future trajectories. Only request trajectory
# columns when the selected input modality actually contains them.
trajectory_features_for_modality = [
    feature for feature in trajectory_feature_names if feature in Features
]

if task not in class_names:
    parser.error(f"Unknown task '{task}'. Choose one of: {', '.join(class_names)}")

required_paths = {
    "processed task data": os.path.join(data_paths["processed_datasets_dir"], task),
}
if include_resnet_features:
    required_paths["ResNet features"] = os.path.join(data_paths["resnet_features_dir"], task)
if include_colin_features:
    required_paths["SpatialCNN features"] = data_paths["spatialcnn_dir"]
if include_segmentation_features:
    required_paths["segmentation features"] = data_paths["segmentation_features_dir"]

missing_paths = [f"{label}: {path}" for label, path in required_paths.items() if not os.path.isdir(path)]
if missing_paths:
    parser.error("Required data directories do not exist:\n  " + "\n  ".join(missing_paths))

os.makedirs(run_results_dir, exist_ok=False)
print(f"Run results directory: {run_results_dir}")

run_config = {
    "timestamp": run_timestamp,
    "slurm_job_id": job_id,
    "task": task,
    "modality": context,
    "model": model_name,
    "model_class": (
        "models.recognition.transtcn.TransformerModel"
        if model_name == "transformer"
        else "models.recognition.compasstcn.TCN"
    ),
    "dataloader": dataloader,
    "num_features": len(Features),
    "feature_names": Features,
    "include_resnet_features": include_resnet_features,
    "include_spatialcnn_features": include_colin_features,
    "include_segmentation_features": include_segmentation_features,
    "recognition_window_mode": dataloader_params["recognition_window_mode"],
    "spatialcnn_split_mode": dataloader_params["spatialcnn_split_mode"],
    "transformer_batch_first": transformer_params.get("batch_first", False),
    "seed_strategy": "gesture_branch_single_process_seed",
    "source_compatibility": "origin/gesture with corrected batch-first attention",
    "configured_model_params": (
        deepcopy(transformer_params)
        if model_name == "transformer"
        else deepcopy(tcn_model_params)
    ),
    "learning_params": learning_params,
    "dataloader_params": dataloader_params,
    "data_paths": data_paths,
}
with open(os.path.join(run_results_dir, "run_config.json"), "w") as outfile:
    json.dump(run_config, outfile, indent=4, default=str)
 
epochs = learning_params["epochs"]
observation_window = dataloader_params["observation_window"],


if(dataloader == "v1"):
    train_dataloader, valid_dataloader = generate_data(dataloader_params["user_left_out"], task, Features, dataloader_params["batch_size"], observation_window, data_paths["processed_datasets_dir"])
elif dataloader == "v2":
    train_dataloader, valid_dataloader = get_dataloaders(tasks=[task],
                                                        subject_id_to_exclude=dataloader_params["user_left_out"],
                                                        observation_window=dataloader_params["observation_window"],
                                                        prediction_window=dataloader_params["prediction_window"],
                                                        batch_size=dataloader_params["batch_size"],
                                                        one_hot=dataloader_params["one_hot"],
                                                        class_names=class_names[task],
                                                        feature_names=Features,
                                                        trajectory_feature_names=trajectory_features_for_modality,
                                                        include_resnet_features=include_resnet_features,
                                                        include_segmentation_features=include_segmentation_features,
                                                        include_colin_features=include_colin_features,
                                                        cast=dataloader_params["cast"],
                                                        normalizer=dataloader_params["normalizer"],
                                                        step=dataloader_params["step"],
                                                        train_sliding_window=False,
                                                        data_paths=data_paths,
                                                        recognition_window_mode=dataloader_params["recognition_window_mode"],
                                                        spatialcnn_split_mode=dataloader_params["spatialcnn_split_mode"])
    # train_dataloader, valid_dataloader = get_dataloaders([task],
    #                                                  dataloader_params["user_left_out"],
    #                                                  dataloader_params["observation_window"],
    #                                                  dataloader_params["prediction_window"],
    #                                                  dataloader_params["batch_size"],
    #                                                  dataloader_params["one_hot"],
    #                                                  class_names = class_names['Suturing'],
    #                                                  feature_names = Features,
    #                                                  include_resnet_features=dataloader_params["include_image_features"],
    #                                                  cast = dataloader_params["cast"],
    #                                                  normalizer = dataloader_params["normalizer"],
    #                                                  step=dataloader_params["step"])

    print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
    print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
    print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

    # loader generator aragement: (src, tgt, future_gesture, future_kinematics)
    print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
    print("Obs Target Shape: ", train_dataloader.dataset[0][1].shape)
    print("Future Target Shape: ", train_dataloader.dataset[0][2].shape)
    print("Future Kinematics Shape: ", train_dataloader.dataset[0][3].shape)
    print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
    print("Train Max Length: ", train_dataloader.dataset.get_max_len())
    print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
    print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
    print("Features: ", train_dataloader.dataset.get_feature_names())

else:
    print("Invalid dataloader choice!")
    exit(-1)

batch = next(iter(train_dataloader))
features = batch[0].shape[-1]
output_dim = batch[1].shape[-1]
input_dim = features  

print("Input Features:",input_dim, "Output Classes:",output_dim)

# Record the effective dimensions after all external modality features have
# been loaded and concatenated. These can differ from len(Features), e.g. for
# ResNet, SpatialCNN, and segmentation modalities.
effective_model_params = (
    deepcopy(transformer_params)
    if model_name == "transformer"
    else deepcopy(tcn_model_params)
)
if model_name == "transformer":
    effective_model_params["encoder_params"]["in_channels"] = input_dim
    effective_model_params["decoder_params"]["out_channels"] = output_dim
    effective_model_params["dropout"] = 0.01

run_config["actual_input_dim"] = input_dim
run_config["output_dim"] = output_dim
run_config["effective_model_params"] = effective_model_params
with open(os.path.join(run_results_dir, "run_config.json"), "w") as outfile:
    json.dump(run_config, outfile, indent=4, default=str)


### Subjects 
subjects = [2,3,4,5,6,7,8,9]
# subjects = [2]


accuracy = []

print("len dataloader:",train_dataloader.dataset.__len__())
# input("Press any key to begin training...")
# Train Loop

REPEAT = 1
for i in range(REPEAT):
    for subject in (subjects):


            model,optimizer,scheduler,criterion = initiate_model(input_dim=input_dim,output_dim=output_dim,transformer_params=transformer_params,learning_params=learning_params, tcn_model_params=tcn_model_params, model_name=model_name)
            
            model.apply(reset_parameters)
            model = model.cuda()
            user_left_out = subject

            if(dataloader == "v1"):
                train_dataloader, valid_dataloader = generate_data(user_left_out, task, Features, dataloader_params["batch_size"], observation_window, data_paths["processed_datasets_dir"])
            else:
                # train_dataloader, valid_dataloader = get_dataloaders([task],
                #                                                 user_left_out,
                #                                                 dataloader_params["observation_window"],
                #                                                 dataloader_params["prediction_window"],
                #                                                 dataloader_params["batch_size"],
                #                                                 dataloader_params["one_hot"],
                #                                                 class_names = class_names['Suturing'],
                #                                                 feature_names = Features,
                #                                                 include_image_features=dataloader_params["include_image_features"],
                #                                                 cast = dataloader_params["cast"],
                #                                                 normalizer = dataloader_params["normalizer"],
                #                                                 step=dataloader_params["step"])
                train_dataloader, valid_dataloader = get_dataloaders(tasks=[task],
                                                        subject_id_to_exclude=user_left_out,
                                                        observation_window=dataloader_params["observation_window"],
                                                        prediction_window=dataloader_params["prediction_window"],
                                                        batch_size=dataloader_params["batch_size"],
                                                        one_hot=dataloader_params["one_hot"],
                                                        class_names=class_names[task],
                                                        feature_names=Features,
                                                        trajectory_feature_names=trajectory_features_for_modality,
                                                        include_resnet_features=include_resnet_features,
                                                        include_segmentation_features=include_segmentation_features,
                                                        include_colin_features=include_colin_features,
                                                        cast=dataloader_params["cast"],
                                                        normalizer=dataloader_params["normalizer"],
                                                        step=dataloader_params["step"],
                                                        train_sliding_window=False,
                                                        data_paths=data_paths,
                                                        recognition_window_mode=dataloader_params["recognition_window_mode"],
                                                        spatialcnn_split_mode=dataloader_params["spatialcnn_split_mode"])
                

            val_loss,acc, all_acc, inference_time, edit_distance, f1_score = traintest_loop(train_dataloader,valid_dataloader,model,optimizer,scheduler,criterion, epochs, dataloader, subject, modality=context, output_dir=run_results_dir)
            
            rolling_avg = rolling_average(all_acc,3)
            # print('Rolling average:',rolling_avg)
            f1_list = list(f1_score.values())
            accuracy.append({'run': i,'subject':subject,  'accuracy':np.max(all_acc), 'rolling_average':rolling_avg[-1], 'edit_score':edit_distance, 'F1@10':f1_list[0], 'F1@25':f1_list[1], 'F1@50':f1_list[2],  'avg_inference_time':inference_time})


if(RECORD_RESULTS):
    
    json_file = 'results'
    with open(os.path.join(run_results_dir, f"{json_file}.json"), "w") as outfile:
        json_object = json.dumps(accuracy, indent=4)
        outfile.write(json_object)

    json_to_csv("summary.csv", json_file, results_dir=run_results_dir)
