from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer


from .dataset import LOUO_Dataset
from .datagen import colin_features, resnet_features, segmentation_features
from .datagen import colin_train_test_splits, colin_features_save_path, segmentation_features_save_path
from .datagen import kinematic_feature_names, kinematic_feature_names_no_ori, trajectory_feature_names, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables
from .dataloader_k import generate_data

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
                    single_window_label: bool = False,
                    train_sliding_window: bool = True,
                    data_paths: dict = None,
                    recognition_window_mode: str = "unified",
                    spatialcnn_split_mode: str = "louo",
                    ):
    
    from typing import List
    import os
    import re
    from functools import partial
    import torch

    from torch.utils.data import DataLoader
    from .dataset import LOUO_Dataset
    from .datagen import all_tasks
    

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


    data_paths = data_paths or {}
    processed_datasets_dir = data_paths.get("processed_datasets_dir", "ProcessedDatasets")
    spatialcnn_dir = data_paths.get("spatialcnn_dir", colin_features_save_path)
    resnet_features_dir = data_paths.get("resnet_features_dir", None)
    segmentation_features_dir = data_paths.get(
        "segmentation_features_dir", segmentation_features_save_path
    )

    # building train and validation datasets and dataloaders
    normalizer = get_normalizer(normalizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files_path, valid_files_path = list(), list()
    for task in tasks:
        data_path = os.path.join(processed_datasets_dir, task)
        tp, vp = _get_files_except_user(task, data_path, subject_id_to_exclude)
        train_files_path += tp
        valid_files_path += vp
    train_kin_files, train_resnet_files = zip(*train_files_path)
    valid_kin_files, valid_resnet_files = zip(*valid_files_path)

    if include_resnet_features and resnet_features_dir:
        train_resnet_files = [
            os.path.join(
                resnet_features_dir,
                os.path.basename(os.path.dirname(kin_path)),
                os.path.basename(feature_path),
            )
            for kin_path, feature_path in zip(train_kin_files, train_resnet_files)
        ]
        valid_resnet_files = [
            os.path.join(
                resnet_features_dir,
                os.path.basename(os.path.dirname(kin_path)),
                os.path.basename(feature_path),
            )
            for kin_path, feature_path in zip(valid_kin_files, valid_resnet_files)
        ]
    elif not include_resnet_features:
        train_resnet_files = []
        valid_resnet_files = []

    colin_features_train, colin_features_valid = [], []
    if include_colin_features:
        if spatialcnn_split_mode == "louo":
            split_number = int(subject_id_to_exclude) - 1
        elif spatialcnn_split_mode == "fixed_split_1":
            split_number = 1
        else:
            raise ValueError(
                "spatialcnn_split_mode must be 'louo' or 'fixed_split_1'; "
                f"received {spatialcnn_split_mode!r}"
            )

        if split_number not in range(1, 9):
            raise ValueError(
                "SpatialCNN features support held-out subjects S02-S09; "
                f"received S{int(subject_id_to_exclude):02d}"
            )

        split_dir = os.path.join(spatialcnn_dir, f"Split_{split_number}")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"SpatialCNN directory for held-out subject "
                f"S{int(subject_id_to_exclude):02d} does not exist: {split_dir}"
            )
        print(
            f"SpatialCNN split mode: {spatialcnn_split_mode}; "
            f"held-out subject: S{int(subject_id_to_exclude):02d}; "
            f"using Split_{split_number}"
        )

        def _spatialcnn_path(kinematics_path: str) -> str:
            filename = os.path.basename(kinematics_path)
            match = re.fullmatch(
                r"(?P<task>.+)_S0?(?P<subject>\d+)_T0?(?P<trial>\d+)\.csv",
                filename,
            )
            if not match:
                raise ValueError(
                    f"Cannot map processed trial filename to SpatialCNN features: {filename}"
                )

            subject = int(match.group("subject"))
            if subject not in range(2, 10):
                raise ValueError(
                    f"SpatialCNN subject mapping supports S02-S09; found S{subject:02d} "
                    f"in {filename}"
                )

            subject_letter = chr(ord("A") + subject - 1)
            feature_filename = (
                f'{match.group("task")}_{subject_letter}'
                f'{int(match.group("trial")):03d}.avi.mat'
            )
            feature_path = os.path.join(split_dir, feature_filename)
            if not os.path.isfile(feature_path):
                raise FileNotFoundError(
                    f"Missing SpatialCNN feature for {filename}: {feature_path}"
                )
            return feature_path

        # Derive features from the already-partitioned kinematic file lists so
        # their order and LOUO membership cannot diverge.
        colin_features_train = [_spatialcnn_path(path) for path in train_kin_files]
        colin_features_valid = [_spatialcnn_path(path) for path in valid_kin_files]

        if len(colin_features_train) != len(train_kin_files):
            raise RuntimeError("SpatialCNN training features are not aligned with trials")
        if len(colin_features_valid) != len(valid_kin_files):
            raise RuntimeError("SpatialCNN validation features are not aligned with trials")

    segmentation_features_train, segmentation_features_valid = [], []
    if include_segmentation_features:
        for file in train_kin_files:
            file_base = os.path.basename(file)
            seg_path = os.path.join(segmentation_features_dir, file_base[9:])
            segmentation_features_train.append(seg_path)
        for file in valid_kin_files:
            file_base = os.path.basename(file)
            seg_path = os.path.join(segmentation_features_dir, file_base[9:])
            segmentation_features_valid.append(seg_path) 
    
    train_dataset = LOUO_Dataset(train_kin_files, observation_window, prediction_window, step=step, onehot=one_hot, class_names=class_names, feature_names=feature_names, trajectory_feature_names=trajectory_feature_names, resnet_files_path=train_resnet_files, colin_files_path=colin_features_train, segmentation_files_path=segmentation_features_train, normalizer=normalizer, sliding_window=train_sliding_window, recognition_window_mode=recognition_window_mode)
    valid_dataset = LOUO_Dataset(valid_kin_files, observation_window, prediction_window, step=step, onehot=one_hot, class_names=class_names, feature_names=feature_names, trajectory_feature_names=trajectory_feature_names, resnet_files_path=valid_resnet_files, colin_files_path=colin_features_valid, segmentation_files_path=segmentation_features_valid, normalizer=normalizer, sliding_window=False, recognition_window_mode=recognition_window_mode)

    target_type = torch.float32 if one_hot else torch.long
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type, cast=cast))
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, collate_fn=partial(LOUO_Dataset.collate_fn, device=device, target_type=target_type, cast=cast)) 

    return train_dataloader, valid_dataloader
