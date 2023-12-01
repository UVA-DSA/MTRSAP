from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer


from dataset import LOUO_Dataset
from datagen import colin_train_test_splits, colin_features_save_path, segmentation_features_save_path

def get_normalizer(normalization_type):
    if normalization_type == 'standardization':
        normalization_object = StandardScaler()
    elif normalization_type == 'min-max':
        normalization_object = MinMaxScaler()
    elif normalization_type == 'power':
        normalization_object = PowerTransformer()
    else:
        normalization_object = None
    return normalization_object

def get_dataloaders(tasks: List[str],
                    subject_id_to_exclude: str,
                    observation_window: int,
                    prediction_window: int,
                    batch_size: int,
                    one_hot: bool,
                    class_names: List[str],
                    feature_names: List[str],
                    trajectory_feature_names: List[str],
                    include_resnet_features: bool,
                    include_segmentation_features: bool,
                    include_colin_features: bool,
                    cast: bool,
                    normalizer: str,
                    step: int = -1,
                    single_window_label: bool = False                    
                    ):
    
    from typing import List
    import os
    from functools import partial
    import torch

    from torch.utils.data import DataLoader
    from dataset import LOUO_Dataset
    from datagen import all_tasks
    

    def _get_files_except_user(task, data_path, subject_id_to_exclude: int) -> List[str]:
        assert(task in all_tasks)
        files = os.listdir(data_path)
        csv_files = [p for p in files if p.endswith(".csv")]
        with open(os.path.join(data_path, "video_feature_files.txt"), 'r') as fp:
            video_files = fp.read().strip().split('\n')
        csv_files.sort(key = lambda x: os.path.basename(x)[:-4])
        video_files.sort(key = lambda x: os.path.basename(x)[:-4])
        except_user = [(os.path.join(data_path, p), v) for (p, v) in zip(csv_files, video_files) if not p.startswith(f"{task}_S0{subject_id_to_exclude}")]
        user = [(os.path.join(data_path, p), v) for (p, v) in zip(csv_files, video_files) if p.startswith(f"{task}_S0{subject_id_to_exclude}")]
        return except_user, user 


    # building train and validation datasets and dataloaders
    normalizer = get_normalizer(normalizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files_path, valid_files_path = list(), list()
    for task in tasks:
        data_path = os.path.join("ProcessedDatasets", task)
        tp, vp = _get_files_except_user(task, data_path, subject_id_to_exclude)
        train_files_path += tp
        valid_files_path += vp
    train_kin_files, train_resnet_files = zip(*train_files_path)
    valid_kin_files, valid_resnet_files = zip(*valid_files_path)

    if not include_resnet_features:
        train_resnet_files = []
        valid_resnet_files = []

    colin_features_train, colin_features_valid = [], []
    if include_colin_features:
        for task in tasks:
            colin_root_for_task = colin_train_test_splits[task][subject_id_to_exclude]
            train_path, test_path = os.path.join(colin_root_for_task, 'train.txt'), os.path.join(colin_root_for_task, 'test.txt')
            train_files = pd.read_csv(train_path, header=None).values.reshape(-1).tolist()
            test_files = pd.read_csv(test_path, header=None).values.reshape(-1).tolist()
            colin_features_train += list(map(lambda x : os.path.join(colin_features_save_path, 'data', task, colin_root_for_task[-7:], x+'.avi.mat'), train_files))
            colin_features_valid += list(map(lambda x : os.path.join(colin_features_save_path, 'data', task, colin_root_for_task[-7:], x+'.avi.mat'), test_files))

    segmentation_features_train, segmentation_features_valid = [], []
    if include_segmentation_features:
        for file in train_kin_files:
            file_base = os.path.basename(file)
            seg_path = os.path.join(segmentation_features_save_path, file_base[9:])
            segmentation_features_train.append(seg_path)
        for file in valid_kin_files:
            file_base = os.path.basename(file)
            seg_path = os.path.join(segmentation_features_save_path, file_base[9:])
            segmentation_features_valid.append(seg_path) 
    
    train_dataset = LOUO_Dataset(train_kin_files, observation_window, prediction_window, step=step, onehot=one_hot, class_names=class_names, feature_names=feature_names, trajectory_feature_names=trajectory_feature_names, resnet_files_path=train_resnet_files, colin_files_path=colin_features_train, segmentation_files_path=segmentation_features_train, normalizer=normalizer, sliding_window=True)
    valid_dataset = LOUO_Dataset(valid_kin_files, observation_window, prediction_window, step=step, onehot=one_hot, class_names=class_names, feature_names=feature_names, trajectory_feature_names=trajectory_feature_names, resnet_files_path=valid_resnet_files, colin_files_path=colin_features_valid, segmentation_files_path=segmentation_features_valid, normalizer=normalizer, sliding_window=False)

    target_type = torch.float32 if one_hot else torch.long
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type, cast=cast))
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type, cast=cast)) 

    return train_dataloader, valid_dataloader  