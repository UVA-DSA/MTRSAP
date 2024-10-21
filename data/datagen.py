import numpy as np
import pandas as pd
import os
import sys

all_tasks = ["Peg_Transfer", "Suturing", "Knot_Tying", "Needle_Passing"]
JIGSAWS_tasks = ["Suturing", "Knot_Tying", "Needle_Passing"]
class_names = {
    "Peg_Transfer": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    "Suturing": ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11'],
    "Knot_Tyring": ['G1', 'G11', 'G12', 'G13', 'G14', 'G15'],
    "Needle_Passing": ["G1", 'G2', 'G3', 'G4', 'G5', 'G6', 'G8', 'G9', 'G10', 'G11']
}
all_class_names = ["G1", 'G2', 'G3', 'G4', 'G5', 'G6',
                   'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15']

# Kinematics feature names
kinematic_feature_names = [
    "PSML_position_x", "PSML_position_y", "PSML_position_z",
    "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
    "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",
    "PSML_gripper_angle",
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
    "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
    "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",
    "PSMR_gripper_angle"
]

kinematic_feature_names_jigsaws = [
    "MTML_position_x", "MTML_position_y", "MTML_position_z", "MTML_rotation_0", "MTML_rotation_1",
    "MTML_rotation_2", "MTML_rotation_3", "MTML_rotation_4", "MTML_rotation_5", "MTML_rotation_6",
    "MTML_rotation_7", "MTML_rotation_8", "MTML_velocity_x", "MTML_velocity_y", "MTML_velocity_z",
    "MTML_velocity_rot0", "MTML_velocity_rot1", "MTML_velocity_rot2", "MTML_gripper_angle",
    "MTMR_position_x", "MTMR_position_y", "MTMR_position_z", "MTMR_rotation_0", "MTMR_rotation_1",
    "MTMR_rotation_2", "MTMR_rotation_3", "MTMR_rotation_4", "MTMR_rotation_5", "MTMR_rotation_6",
    "MTMR_rotation_7", "MTMR_rotation_8", "MTMR_velocity_x", "MTMR_velocity_y", "MTMR_velocity_z",
    "MTMR_velocity_rot0", "MTMR_velocity_rot1", "MTMR_velocity_rot2", "MTMR_gripper_angle",
    "PSML_position_x", "PSML_position_y", "PSML_position_z", "PSML_rotation_0", "PSML_rotation_1",
    "PSML_rotation_2", "PSML_rotation_3", "PSML_rotation_4", "PSML_rotation_5", "PSML_rotation_6",
    "PSML_rotation_7", "PSML_rotation_8", "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
    "PSML_velocity_rot0", "PSML_velocity_rot1", "PSML_velocity_rot2", "PSML_gripper_angle",
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z", "PSMR_rotation_0", "PSMR_rotation_1",
    "PSMR_rotation_2", "PSMR_rotation_3", "PSMR_rotation_4", "PSMR_rotation_5", "PSMR_rotation_6",
    "PSMR_rotation_7", "PSMR_rotation_8", "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
    "PSMR_velocity_rot0", "PSMR_velocity_rot1", "PSMR_velocity_rot2", "PSMR_gripper_angle"
]

# Patient position-related kinematics feature names
kinematic_feature_names_jigsaws_patient_position = [
    "PSML_position_x", "PSML_position_y", "PSML_position_z",
    "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
    "PSML_gripper_angle",
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
    "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
    "PSMR_gripper_angle"
]

# Trajectory feature names
trajectory_feature_names = [
    "PSML_position_x", "PSML_position_y", "PSML_position_z",
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z"
]

# Context features
state_variables = ['left_holding', 'left_contact', 'right_holding', 'right_contact', 'needle_state']
state_variables_repeating_factor = 10

# ResNet features
resnet_features_save_path = './resnet_features'
resnet_features = [f'resnet_{i}' for i in range(2048)]

# Colin features
colin_features = [f'colin_{i}' for i in range(128)]
colin_features_save_path = './SpatialCNN/'
colin_train_test_splits = {
    'Suturing': {
        2: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_1"),
        3: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_2"),
        4: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_3"),
        5: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_4"),
        6: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_5"),
        7: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_6"),
        8: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_7"),
        9: os.path.join(colin_features_save_path, "splits", "JIGSAWS", "Suturing", "Split_8"),
    }
}

# Segmentation features
segmentation_features = [f'seg_{i}' for i in range(128)]
segmentation_features_save_path = "./segmentation_masks/pca_features_normalized"
image_features_save_path = "./resnet_features/"


# New function for feature name mapping
def map_feature_names(feature_names):
    """
    Maps column names from MTML_* to PSML_* to match expected feature names.

    Parameters:
    - feature_names (list): List of column names from the dataset.

    Returns:
    - mapped_names (list): Updated list of column names with PSML_ prefix.
    """
    mapped_names = []
    for name in feature_names:
        if name.startswith('MTML_'):
            mapped_names.append(name.replace('MTML_', 'PSML_'))
        else:
            mapped_names.append(name)
    return mapped_names


def load_dataset(file_path):
    """
    Loads the dataset from a given CSV file and applies feature name mapping.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - df (DataFrame): Loaded and processed DataFrame.
    """
    df = pd.read_csv(file_path)

    # Apply feature name mapping
    df.columns = map_feature_names(df.columns)

    return df


def generate_data(task: str):
    """
    Generate data for a given task.

    Parameters:
    - task (str): Name of the task to process.
    """
    processed_data_path = "./ProcessedDatasets"
    root_path = os.path.join("./Datasets", "dV")
    task_path = os.path.join(root_path, task)
    task_path_target = os.path.join(processed_data_path, task)
    gestures_path = os.path.join(task_path, "gestures")
    video_path = os.path.join(task_path, "video")
    kinematics_path = os.path.join(task_path, "kinematics")
    transcriptions_path = os.path.join(task_path, "transcriptions")

    # Ensure directories exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    if not os.path.exists(task_path_target):
        os.makedirs(task_path_target)

    videos = []
    print(f"Looking for gesture files in: {gestures_path}")  # A print statement to confirm the exact value of gestures_path:
    for file in os.listdir(gestures_path):
        if file.startswith(task):
            # Read the gesture labels
            file_path = os.path.join(gestures_path, file)
            labels = pd.read_csv(file_path, sep=' ', index_col=None, header=None)
            if len(labels.columns) == 5:
                column_names = ['Start', "Stop", "MP", "Success", "X"]
                labels = labels.set_axis(column_names, axis=1).drop("X", axis=1)
            else:
                column_names = ['Start', "Stop", "MP", "Success"]
                labels = labels.set_axis(column_names, axis=1)

            # Read kinematic variables
            kinematics = pd.read_csv(os.path.join(kinematics_path, file[:-3] + 'csv'), index_col=None)

            # Apply feature name mapping to kinematics
            kinematics.columns = map_feature_names(kinematics.columns)

            # Read state variables
            states_df = pd.read_csv(os.path.join(transcriptions_path, file), sep=' ', index_col=0, header=None)
            if len(states_df.columns) == 6:
                column_names = [*state_variables, "X"]
                states_df = states_df.set_axis(column_names, axis=1).drop("X", axis=1)
            else:
                column_names = state_variables
                states_df = states_df.set_axis(column_names, axis=1)

            # Concatenate states with kinematics
            kinematics = pd.concat([kinematics, states_df], axis=1)
            kinematics = kinematics.ffill(axis=0)

            # Add labels to the kinematics
            kinematics['label'] = ['-'] * len(kinematics)
            for i, row in labels.iterrows():
                start, stop, mp = int(row['Start']), int(row['Stop']), row['MP']
                kinematics.loc[start:stop, 'label'] = mp

            # Save the processed file
            kinematics.to_csv(os.path.join(task_path_target, file[:-3] + 'csv'), index=False)

            # Collect video feature paths
            video_features_file_path = os.path.join(resnet_features_save_path, task, file[:-4] + '_Right' + '.npy')
            if not os.path.exists(video_features_file_path):
                video_features_file_path = os.path.join(resnet_features_save_path, task, file[:-4] + '_Left' + '.npy')
            if not os.path.exists(video_features_file_path):
                raise ValueError(f"Features for video file {os.path.basename(video_features_file_path)} do not exist.")
            videos.append(video_features_file_path)

    # Save video feature paths to a text file
    with open(os.path.join(task_path_target, 'video_feature_files.txt'), 'w') as fp:
        for v in videos:
            fp.write(v + '\n')


if __name__ == "__main__":
    task = sys.argv[1]
    assert task in all_tasks, f"Task '{task}' is not recognized. Must be one of {all_tasks}."
    generate_data(task)