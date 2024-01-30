import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict

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

kinematic_feature_names = ["PSML_position_x", "PSML_position_y", "PSML_position_z",
                           "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
                           "PSML_orientation_x", "PSML_orientation_y", "PSML_orientation_z", "PSML_orientation_w",
                           "PSML_gripper_angle",
                           "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
                           "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
                           "PSMR_orientation_x", "PSMR_orientation_y", "PSMR_orientation_z", "PSMR_orientation_w",
                           "PSMR_gripper_angle"]
kinematic_feature_names_jigsaws = ["MTML_position_x", "MTML_position_y", "MTML_position_z",
                                   "MTML_rotation_0", "MTML_rotation_1", "MTML_rotation_2", "MTML_rotation_3", "MTML_rotation_4", "MTML_rotation_5", "MTML_rotation_6", "MTML_rotation_7", "MTML_rotation_8",
                                   "MTML_velocity_x", "MTML_velocity_y", "MTML_velocity_z",
                                   "MTML_velocity_rot0", "MTML_velocity_rot1", "MTML_velocity_rot2",
                                   "MTML_gripper_angle",
                                   "MTMR_position_x", "MTMR_position_y", "MTMR_position_z",
                                   "MTMR_rotation_0", "MTMR_rotation_1", "MTMR_rotation_2", "MTMR_rotation_3", "MTMR_rotation_4", "MTMR_rotation_5", "MTMR_rotation_6", "MTMR_rotation_7", "MTMR_rotation_8",
                                   "MTMR_velocity_x", "MTMR_velocity_y", "MTMR_velocity_z",
                                   "MTMR_velocity_rot0", "MTMR_velocity_rot1", "MTMR_velocity_rot2",
                                   "MTMR_gripper_angle",
                                   "PSML_position_x", "PSML_position_y", "PSML_position_z",
                                   "PSML_rotation_0", "PSML_rotation_1", "PSML_rotation_2", "PSML_rotation_3", "PSML_rotation_4", "PSML_rotation_5", "PSML_rotation_6", "PSML_rotation_7", "PSML_rotation_8",
                                   "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
                                   "PSML_velocity_rot0", "PSML_velocity_rot1", "PSML_velocity_rot2",
                                   "PSML_gripper_angle",
                                   "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
                                   "PSMR_rotation_0", "PSMR_rotation_1", "PSMR_rotation_2", "PSMR_rotation_3", "PSMR_rotation_4", "PSMR_rotation_5", "PSMR_rotation_6", "PSMR_rotation_7", "PSMR_rotation_8",
                                   "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
                                   "PSMR_velocity_rot0", "PSMR_velocity_rot1", "PSMR_velocity_rot2",
                                   "PSMR_gripper_angle"]

kinematic_feature_names_jigsaws_patient_position = ["PSML_position_x", "PSML_position_y", "PSML_position_z",
                                                    "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
                                                    "PSML_gripper_angle",
                                                    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
                                                    "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
                                                    "PSMR_gripper_angle"]

colin_features = ["colin_0", "colin_1", "colin_2", "colin_3", "colin_4", "colin_5", "colin_6", "colin_7", "colin_8", "colin_9", "colin_10", "colin_11", "colin_12", "colin_13", "colin_14", "colin_15", "colin_16", "colin_17", "colin_18", "colin_19", "colin_20", "colin_21", "colin_22", "colin_23", "colin_24", "colin_25", "colin_26", "colin_27", "colin_28", "colin_29", "colin_30", "colin_31", "colin_32", "colin_33", "colin_34", "colin_35", "colin_36", "colin_37", "colin_38", "colin_39", "colin_40", "colin_41", "colin_42", "colin_43", "colin_44", "colin_45", "colin_46", "colin_47", "colin_48", "colin_49", "colin_50", "colin_51", "colin_52", "colin_53", "colin_54", "colin_55", "colin_56", "colin_57", "colin_58", "colin_59", "colin_60", "colin_61", "colin_62", "colin_63", "colin_64", "colin_65", "colin_66", "colin_67", "colin_68", "colin_69", "colin_70", "colin_71", "colin_72", "colin_73", "colin_74", "colin_75", "colin_76", "colin_77", "colin_78", "colin_79", "colin_80", "colin_81", "colin_82", "colin_83", "colin_84", "colin_85", "colin_86", "colin_87", "colin_88", "colin_89", "colin_90", "colin_91", "colin_92", "colin_93", "colin_94", "colin_95", "colin_96", "colin_97", "colin_98", "colin_99", "colin_100", "colin_101", "colin_102", "colin_103", "colin_104", "colin_105", "colin_106", "colin_107", "colin_108", "colin_109", "colin_110", "colin_111", "colin_112", "colin_113", "colin_114", "colin_115", "colin_116", "colin_117", "colin_118", "colin_119", "colin_120", "colin_121", "colin_122", "colin_123", "colin_124", "colin_125", "colin_126", "colin_127"]
state_variables = ['left_holding', 'left_contact',
                   'right_holding', 'right_contact', 'needle_state']

img_features = ["resnet_" + str(i) for i in range(2048)]

segmentation_features = ["seg_" + str(i) for i in range(128)]

state_variables_repeating_factor = 10

image_features_save_path = './ImageFeatures'


def generate_data(task: str):
    processed_data_path = "./ProcessedDatasets"
    root_path = os.path.join("./Datasets", "dV")
    task_path = os.path.join(root_path, task)
    task_path_target = os.path.join(processed_data_path, task)
    gestures_path = os.path.join(task_path, "gestures")
    video_path = os.path.join(task_path, "video")
    kinematics_path = os.path.join(task_path, "kinematics")
    transcriptions_path = os.path.join(task_path, "transcriptions")

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    if not os.path.exists(task_path_target):
        os.makedirs(task_path_target)

    videos = []
    for file in os.listdir(gestures_path):
        if file.startswith(task):

            # read the gesture labels
            file_path = os.path.join(gestures_path, file)
            labels = pd.read_csv(file_path, sep=' ',
                                 index_col=None, header=None)
            if len(labels.columns) == 5:
                column_names = ['Start', "Stop", "MP", "Success", "X"]
                labels = labels.set_axis(
                    column_names, axis=1).drop("X", axis=1)
            else:
                column_names = ['Start', "Stop", "MP", "Success"]
                labels = labels.set_axis(column_names, axis=1)

            # read kinematic variables
            kinematics = pd.read_csv(os.path.join(
                kinematics_path, file[:-3] + 'csv'), index_col=None)

            # read state variables
            states_df = pd.read_csv(os.path.join(
                transcriptions_path, file), sep=' ', index_col=0, header=None)
            if len(states_df.columns) == 6:
                column_names = [*state_variables, "X"]
                states_df = states_df.set_axis(
                    column_names, axis=1).drop("X", axis=1)
            else:
                column_names = state_variables
                states_df = states_df.set_axis(column_names, axis=1)

            # append states to the kinematics
            kinematics = pd.concat([kinematics, states_df], axis=1)
            kinematics = kinematics.ffill(axis=0)

            # append labels to the kinematics
            kinematics['label'] = ['-'] * len(kinematics)
            for i, row in labels.iterrows():
                start, stop, mp = int(row['Start']), int(
                    row['Stop']), row['MP']
                kinematics.loc[start:stop, 'label'] = mp

            # save compound file contaning kinematics, state variables and gesture label
            kinematics.to_csv(os.path.join(task_path_target,
                              file[:-3] + 'csv'), index=False)

            # collect the video path files
            video_features_file_path = os.path.join(
                image_features_save_path, task, file[:-4] + '_Right' + '.npy')
            if not os.path.exists(video_features_file_path):
                video_features_file_path = os.path.join(
                    image_features_save_path, task, file[:-4] + '_Left' + '.npy')
            if not os.path.exists(video_features_file_path):
                raise ValueError(
                    f"The features for video file {os.path.basename(video_features_file_path)} does not exist")
            videos.append(video_features_file_path)

    with open(os.path.join(task_path_target, 'video_feature_files.txt'), 'w') as fp:
        for v in videos:
            fp.write(v + '\n')


if __name__ == "__main__":
    task = sys.argv[1]
    assert (task in all_tasks)
    generate_data(task)
