import torch
import numpy as np
import json
from data import get_dataloaders, generate_data
from config import *
from models.utils import reset_parameters, traintest_loop, rolling_average
from models import initiate_model
from utils import json_to_csv

import datetime
import argparse


torch.manual_seed(0)


# end of imports #


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A simple command-line argument parser")

# Add arguments
parser.add_argument("--model", help="Specify which model to run", required=True)
parser.add_argument("--dataloader", help="Specify which dataloader", required=True)
parser.add_argument("--modality", help="Specify which modality combo", required=True, type=int)
# parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
model_name = args.model
dataloader = args.dataloader
context = args.modality
# verbose_mode = args.verbose


# manual seeding ensure reproducibility
# torch.manual_seed(0)



# tasks and features to be included
task = "Suturing"


# context = dataloader_params["context"]

if context in modality_mapping:
    Features = modality_mapping[context]
else:
    print("Invalid modality choice!")
    exit(-1)
 
epochs = learning_params["epochs"]
observation_window = dataloader_params["observation_window"],


if(dataloader == "v1"):
    train_dataloader, valid_dataloader = generate_data(dataloader_params["user_left_out"],task,Features, dataloader_params["batch_size"], observation_window)
elif dataloader == "v2":
    # train_dataloader, valid_dataloader = get_dataloaders(tasks=[task],
    #                                                     subject_id_to_exclude=dataloader_params["user_left_out"],
    #                                                     observation_window=dataloader_params["observation_window"],
    #                                                     prediction_window=dataloader_params["prediction_window"],
    #                                                     batch_size=dataloader_params["batch_size"],
    #                                                     one_hot=one_hot,
    #                                                     class_names=class_names['Suturing'],
    #                                                     feature_names=Features,
    #                                                     trajectory_feature_names=trajectory_feature_names,
    #                                                     include_resnet_features=include_resnet_features,
    #                                                     include_segmentation_features=include_segmentation_features,
    #                                                     include_colin_features=include_colin_features,
    #                                                     cast=cast,
    #                                                     normalizer=normalizer,
    #                                                     step=step)
    train_dataloader, valid_dataloader = get_dataloaders([task],
                                                     dataloader_params["user_left_out"],
                                                     dataloader_params["observation_window"],
                                                     dataloader_params["prediction_window"],
                                                     dataloader_params["batch_size"],
                                                     dataloader_params["one_hot"],
                                                     class_names = class_names['Suturing'],
                                                     feature_names = Features,
                                                     include_resnet_features=dataloader_params["include_image_features"],
                                                     cast = dataloader_params["cast"],
                                                     normalizer = dataloader_params["normalizer"],
                                                     step=dataloader_params["step"])

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


### DEFINE MODEL HERE ###
# model_name = 'tcn' 
# model_name = 'transformer'

model,optimizer,scheduler,criterion = initiate_model(input_dim=input_dim,output_dim=output_dim,transformer_params=transformer_params,learning_params=learning_params, tcn_model_params=tcn_model_params, model_name=model_name)

print(model)


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
                train_dataloader, valid_dataloader = generate_data(user_left_out,task,Features, dataloader_params["batch_size"], observation_window)
            else:
                train_dataloader, valid_dataloader = get_dataloaders([task],
                                                                user_left_out,
                                                                dataloader_params["observation_window"],
                                                                dataloader_params["prediction_window"],
                                                                dataloader_params["batch_size"],
                                                                dataloader_params["one_hot"],
                                                                class_names = class_names['Suturing'],
                                                                feature_names = Features,
                                                                include_image_features=dataloader_params["include_image_features"],
                                                                cast = dataloader_params["cast"],
                                                                normalizer = dataloader_params["normalizer"],
                                                                step=dataloader_params["step"])
                

            val_loss,acc, all_acc, inference_time, edit_distance, f1_score = traintest_loop(train_dataloader,valid_dataloader,model,optimizer,scheduler,criterion, epochs, dataloader, subject, modality=context)
            
            rolling_avg = rolling_average(all_acc,3)
            # print('Rolling average:',rolling_avg)
            f1_list = list(f1_score.values())
            accuracy.append({'run': i,'subject':subject,  'accuracy':np.max(all_acc), 'rolling_average':rolling_avg[-1], 'edit_score':edit_distance, 'F1@10':f1_list[0], 'F1@25':f1_list[1], 'F1@50':f1_list[2],  'avg_inference_time':inference_time})


if(RECORD_RESULTS):
    
    json_file = 'train_results'
    with open(f"./results/{json_file}.json", "w") as outfile:
        json_object = json.dumps(accuracy, indent=4)
        outfile.write(json_object)
        

    current_datetime = datetime.datetime.now()

    # Format the datetime as a string to be used as a filename
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    csv_name = f'Train_{task}_{model_name}_{formatted_datetime}_MODALITY_{context}_num_features{len(Features)}_LOUO_window{dataloader_params["observation_window"]}.csv'
         
    json_to_csv(csv_name, json_file)    