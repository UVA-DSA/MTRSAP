from typing import List, Tuple
import copy
import torch
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.io as sio

import math



def normalize_columns(matrix, normalizer):

    """
    Normalize specified columns of a numpy matrix.

    Parameters:
    - normalization_object: An optional normalization object; if None, an object will be learned from the data
    - matrix (numpy.ndarray): Input matrix to be normalized.
    - normalization_type (str): Type of normalization to apply ('standardization', 'min-max', 'power').
    - columns_to_normalize (list or None): Indices of columns to be normalized. If None, normalize all columns.

    Returns:
    - normalized_matrix (numpy.ndarray): Normalized matrix.
    - normalization_object: Object used for normalization, can be used for future data using the same normalization.
    """


    normalized_matrix = matrix.copy()

    if hasattr(normalizer, 'n_features_in_'):
        print(f"Normalizing the data; Method: {str(normalizer.__class__()).lower()}")
        normalized_matrix = normalizer.transform(normalized_matrix)
    else:
        # Fit and transform the specified columns
        print(f"Learning and normalizing the data; Method: {str(normalizer.__class__()).lower()}")
        normalized_matrix = normalizer.fit_transform(normalized_matrix)

    return normalized_matrix, normalizer



class LOUO_Dataset(Dataset):
    
    def __init__(self,
                kin_files_path: List[str],
                observation_window_size: int,
                prediction_window_size: int,
                step: int = 0,
                onehot: bool = False,
                class_names: List[str] = [],
                feature_names: List[str] = [],
                trajectory_feature_names: List[str] = [],
                resnet_files_path: List[str] = [],
                colin_files_path: List[str] = [],
                segmentation_files_path: List[str] = [],
                normalizer: object = None, # (normalization_object of the type['standardization', 'min-max', 'power'])
                single_window_label: bool = False, # instead of frame-wise labels, return a single label for a window
                sliding_window: bool = True, # slide the window by 1 timestep, or jump the window by `observation_window`
            ):
        
        self.kin_files_path = kin_files_path
        self.resnet_files_path = resnet_files_path
        self.colin_files_path = colin_files_path
        self.segmentation_files_path = segmentation_files_path
        self.observation_window_size = observation_window_size
        self.prediction_window_size = prediction_window_size
        self.target_names = class_names
        self.feature_names_ = copy.deepcopy(feature_names)
        self.sliding_window = sliding_window
        self.le = preprocessing.LabelEncoder()
        self.onehot = onehot
        if onehot:
            self.enc = preprocessing.OneHotEncoder(sparse_output=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = step
        self.single_window_label = single_window_label

        # reading the data (kinematic features, [image features, context features])
        (self.X, self.Y) = self._load_data() 
        self.feature_names = self.feature_names_

        self.traj_features_indices = [self.feature_names.index(feature) for feature in trajectory_feature_names]

        # feature normalization
        if normalizer:
            self.X, self.normalizer = normalize_columns(self.X, normalizer)

    def get_feature_names(self):
        return self.feature_names
    
    def get_target_names(self):
        return self.target_names
    
    def get_num_trials(self):
        return len(self.samples_per_trial)
    
    def get_max_len(self):
        return self.max_len
    
    def get_class_weights(self):
        value_counts = pd.Series(self.Y).value_counts(normalize=True).sort_index()
        value_counts = 1./value_counts.to_numpy()
        return value_counts/value_counts.sum()
    
    def get_trial(self, trial_id: int, window_size: int = -1):
        if trial_id == 0:
            trial_start = 0
        else:
            trial_start = sum(self.samples_per_trial[:trial_id])
        trial_end = trial_start + self.samples_per_trial[trial_id]
        print(trial_start, trial_end, self.X.shape)
        _X = self.X[trial_start + 1 : trial_end + 1]
        _Y = self.Y[trial_start : trial_end + 1]
        if window_size > 0:
            X_image = []
            X = []
            Y = []
            Y_future = []
            P = []
            num_windows = _X.shape[0]//window_size
            for i in range(num_windows - 1):
                X.append(_X[i*window_size:(i+1)*window_size])
                Y.append(_Y[i*window_size:(i+1)*window_size])
                Y_future.append(_Y[(i+1)*window_size:(i+2)*window_size])
                P.append(_X[(i+1)*window_size:(i+2)*window_size, self.traj_features_indices])
            X = np.array(X)
            Y = np.array(Y)
            Y_future = np.array(Y_future)
            P = np.array(P)
        else:
            X = _X
            Y = _Y
            Y_future = np.array([])
            P = np.array([])
        X = torch.from_numpy(X).to(torch.float32).to(self.device) # shape [num_windows, window_size, features_dim]
        target_type = target_type = torch.float32 if self.onehot else torch.long
        Y = torch.from_numpy(Y).to(target_type).to(self.device)
        Y_future = torch.from_numpy(Y_future).to(target_type).to(self.device)
        P = torch.from_numpy(P).to(torch.float32).to(self.device)
        return X, Y, Y_future, P
    
    def _load_data(self):
        X = []
        X_image = []
        Y = []
        self.samples_per_trial = []
        
        for i, kinematics_path in enumerate(self.kin_files_path):
            if os.path.isfile(kinematics_path) and kinematics_path.endswith('.csv'):
                kinematics_data = pd.read_csv(kinematics_path)

                feature_names = kinematics_data.columns.to_list()[:-1] if not self.feature_names_ else self.feature_names_
                kin_data = kinematics_data.loc[:, feature_names]
                kin_label = kinematics_data.iloc[:,-1] # last column is always taken to be the target class

                if self.resnet_files_path:
                    resnet_features_path = self.resnet_files_path[i]
                    resnet_features = np.load(resnet_features_path)

                if self.colin_files_path:
                    colin_features_path = self.colin_files_path[i]
                    colin_features = sio.loadmat(colin_features_path)['A']
                    colin_features = np.repeat(colin_features, repeats=3, axis=0)

                if self.segmentation_files_path:
                    segmentation_features = pd.read_csv(self.segmentation_files_path[i], header=None)
                    segmentation_features = np.repeat(segmentation_features, repeats=3, axis=0)

                if self.step > 1:
                    if self.colin_files_path:
                        assert self.step in [3, 6], 'When using colin features, which is computed in 10 Hz, you must choose step=3 (10Hz) or 6(5Hz)'
                        if self.step == 6:
                            colin_features = colin_features[::2]
                    kin_data = kin_data.loc[pd.RangeIndex(start=0, stop=len(kin_data), step=self.step)].reset_index(drop=True)
                    kin_label = kin_label.loc[pd.RangeIndex(start=0, stop=len(kin_label), step=self.step)].reset_index(drop=True)
                    if self.resnet_files_path:
                        resnet_features = resnet_features[::self.step]
                    if self.segmentation_files_path:
                        segmentation_features = segmentation_features[::self.step]
                
                # limit by the length of the smaller tensor, image or kin
                last_index = len(kin_data)
                if self.resnet_files_path:
                    last_index = min(last_index, resnet_features.shape[0])
                if self.colin_files_path:
                    last_index = min(last_index, colin_features.shape[0])
                kin_data = kin_data.iloc[:last_index]
                kin_label = kin_label.iloc[:last_index]
                if self.resnet_files_path:
                    resnet_features = resnet_features[:last_index]
                if self.colin_files_path:
                    colin_features = colin_features[:last_index]
                if self.segmentation_files_path:
                    segmentation_features = segmentation_features[:last_index]

                # drop the frames where the label does not exist
                drop_ind = kin_label[kin_label == '-'].index
                kin_data = kin_data.drop(index=drop_ind)
                kin_label = kin_label.drop(index=drop_ind)
                if self.resnet_files_path:
                    resnet_features = np.delete(resnet_features, drop_ind.tolist(), axis=0)
                if self.colin_files_path:
                    colin_features = np.delete(colin_features, drop_ind.tolist(), axis=0)
                if self.segmentation_files_path:
                    segmentation_features = np.delete(segmentation_features, drop_ind.tolist(), axis=0)
                
                X_image_subject = []

                if self.resnet_files_path:
                    X_image_subject.append(resnet_features)
                if self.colin_files_path:
                    X_image_subject.append(colin_features)
                if self.segmentation_files_path:
                    X_image_subject.append(segmentation_features)
                if len(X_image_subject) > 0:
                    X_image_subject = np.concatenate(X_image_subject, axis=1)

                if len(X_image_subject) > 1:
                    X_image.append(X_image_subject)

                X.append(kin_data.values)
                Y.append(kin_label.values)
                self.samples_per_trial.append(len(kin_data))

        self.max_len = max([d.shape[0] for d in X])
        
        # label encoding and transformation
        if not self.target_names:
            self.le.fit(np.concatenate(Y))
        else:
            self.le.fit(np.array(self.target_names))
        self.target_names = self.le.classes_
        print(self.target_names)
        Y = [self.le.transform(yi) for yi in Y]

        # one-hot encoding
        if self.onehot:
            self.enc.fit(np.arange(len(self.target_names)).reshape(-1, 1))
            Y = [yi.reshape(len(yi), 1) for yi in Y]
            Y = [self.enc.transform(yi) for yi in Y]
        
        # store data inside a single 2D numpy array
        X = np.concatenate(X)
        if X_image:
            X_image = np.concatenate(X_image)
            X = np.concatenate([X, X_image], axis=1)
        if self.resnet_files_path:
            self.feature_names_ += [f'resnet_{i}' for i in range(resnet_features.shape[1])]
        if self.colin_files_path:
            self.feature_names_ += [f'colin_{i}' for i in range(colin_features.shape[1])]
        if self.segmentation_files_path:
            self.feature_names_ += [f'seg_{i}' for i in range(segmentation_features.shape[1])]
        Y = np.concatenate(Y)

        return X, Y
    
    def __len__(self):
        # this should return the size of the dataset
        len_ =  self.Y.shape[0] - self.observation_window_size - self.prediction_window_size
        if not self.sliding_window:
            len_ = math.floor(len_//self.observation_window_size)
        return len_
    
    # this should return one sample from the dataset
    def __getitem__(self, idx):
        start_idx = idx + 1 if self.sliding_window else idx*self.observation_window_size + 1
        end_index = start_idx + self.observation_window_size
        
        features = self.X[start_idx : end_index]
        target = self.Y[start_idx-1 : end_index] # one additional observation is given for recursive decoding in recognition task
        gesture_pred_target = self.Y[end_index : end_index + self.prediction_window_size]
        traj_pred_target = self.X[end_index : end_index + self.prediction_window_size, self.traj_features_indices]
        
        # return kinematic_features, image_features, target, gesture_pred_target, traj_pred_target
        return features, target, gesture_pred_target, traj_pred_target
    
    def __get_dominant_label(self, y):
        pass

    
    @staticmethod
    def collate_fn(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), target_type=torch.float32, cast: bool = True):
        X = []
        Y = []
        Future_Y = []
        P = []
        for x, y, yy, p in batch:
            X.append(x)
            Y.append(y)
            Future_Y.append(yy)
            P.append(p)
        X = np.array(X)
        Y = np.array(Y)
        try:
            Future_Y = np.array(Future_Y)
            P = np.array(P)
        except:
            Future_Y = np.array(Future_Y[:-1])
            P = np.array(P[:-1])

        if cast:
            # cast to torch tensor
            x_batch = torch.from_numpy(X)
            y_batch = torch.from_numpy(Y)
            yy_batch = torch.from_numpy(Future_Y)
            p_batch = torch.from_numpy(P)
            
            # cast to appropriate data type
            x_batch = x_batch.to(torch.float32)
            y_batch = y_batch.to(target_type)
            yy_batch = yy_batch.to(target_type)
            p_batch = p_batch.to(torch.float32)

            # cast to appropriate device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            yy_batch = yy_batch.to(device)
            p_batch = p_batch.to(device)
        else:
            x_batch = X
            y_batch = Y
            yy_batch = Future_Y
            p_batch = P

        return (x_batch, y_batch, yy_batch, p_batch)


                