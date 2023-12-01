import os
import shutil

import pandas as pd

from datagen import kinematic_feature_names_jigsaws


def create_nested_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists. Skipping creation.")

if __name__ == '__main__':

    # task = input("Please specify the task: ")
    task = 'Suturing'
    subject_maps = dict(
        B='S02',
        C='S03',
        D='S04',
        E='S05',
        F='S06',
        G='S07',
        H='S08',
        I='S09'
    )
    video_map = dict(capture1='Left', capture2='Right')

    # make `Datasets` like paths if they don't exist already
    Datasets_path = './Datasets/dV'
    dest_task_path = os.path.join(Datasets_path, task)
    dest_kin_folder = os.path.join(dest_task_path, 'kinematics')
    dest_video_folder = os.path.join(dest_task_path, 'video')
    dest_gesture_folder = os.path.join(dest_task_path, 'gestures')
    create_nested_directory(Datasets_path)
    create_nested_directory(dest_kin_folder)
    create_nested_directory(dest_video_folder)
    create_nested_directory(dest_gesture_folder)

    task_path = os.path.join('JIGSAWS_Dataset', 'JIGSAWS_Dataset', task, task)
    kin_path = os.path.join(task_path, 'kinematics', 'AllGestures')
    video_path = os.path.join(task_path, 'video')
    transcriptions_path = os.path.join(task_path, 'transcriptions')

    # iterate through files, get the currect name and dest path for a file, and create/copy the correct file

    # kinematics
    for file in os.listdir(kin_path):
        if file.endswith('.txt'):
            for letter in subject_maps.keys():
                if letter in file:
                    dest_file = file.replace(letter+'0', subject_maps[letter]+'_T').replace('.txt', '.csv')
            file = os.path.join(kin_path, file)
            dest_file = os.path.join(dest_kin_folder, dest_file)
            kin_data = pd.read_csv(file, delimiter='\s+', on_bad_lines='skip', header=None)
            kin_data.columns = kinematic_feature_names_jigsaws
            kin_data.to_csv(dest_file, index=False)

# video
for file in os.listdir(video_path):
    if file.endswith('.avi'):
        for letter in subject_maps.keys():
            if letter in file:
                dest_file = file.replace(letter+'0', subject_maps[letter]+'_T')
        for c in video_map:
            if c in file:
                dest_file = dest_file.replace(c, video_map[c])
        file = os.path.join(video_path, file)
        dest_file = os.path.join(dest_video_folder, dest_file)
        shutil.copy(file, dest_file)

    
# gestures
for file in os.listdir(transcriptions_path):
    if file.endswith('.txt'):
        for letter in subject_maps.keys():
            if letter in file:
                dest_file = file.replace(letter+'0', subject_maps[letter]+'_T')
        file = os.path.join(transcriptions_path, file)
        dest_file = os.path.join(dest_gesture_folder, dest_file)
        shutil.copy(file, dest_file)


