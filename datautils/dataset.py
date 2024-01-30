from typing import List, Tuple
import torch
import os
from sklearn import preprocessing
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



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
                files_path: List[str],
                observation_window_size: int,
                prediction_window_size: int,
                step: int = 0,
                onehot: bool = False,
                class_names: List[str] = [],
                feature_names: List[str] = [],
                include_image_features: bool = False,
                normalizer: object = None # (normalization_object of the type['standardization', 'min-max', 'power'])
            ):
        
        self.files_path = files_path
        self.observation_window_size = observation_window_size
        self.prediction_window_size = prediction_window_size
        self.target_names = class_names
        self.feature_names_ = feature_names
        self.le = preprocessing.LabelEncoder()
        self.onehot = onehot
        if onehot:
            self.enc = preprocessing.OneHotEncoder(sparse_output=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = step

        # reading the data (kinematic features, [image features, context features])
        (self.X, _X_image, self.Y) = self._load_data() 
        self.feature_names = self.feature_names_
        if include_image_features:
            self.X = np.concatenate([self.X, _X_image], axis=-1)
            self.feature_names = self.feature_names_ + [f"img_{i}" for i in range(_X_image.shape[-1])]
        
        if step > 0:
            self.X, self.Y = self.X[::step], self.Y[::step] # resampling the data (e.g. in going from 30Hz to 10Hz, set step=3)

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
    
    def get_trial(self, trial_id: int, window_size: int = -1):
        if trial_id == 0:
            trial_start = 0
        else:
            trial_start = sum(self.samples_per_trial[:trial_id])
        trial_end = trial_start + self.samples_per_trial[trial_id]
        _X = self.X[trial_start + 1 : trial_end + 1]
        print(_X.shape)
        # _X_image = self.X_image[trial_start + 1 : trial_end + 1]
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
                # X_image.append(_X_image[i*window_size:(i+1)*window_size])
                Y.append(_Y[i*window_size:(i+1)*window_size])
                Y_future.append(_Y[(i+1)*window_size:(i+2)*window_size])
                P.append(_X[(i+1)*window_size:(i+2)*window_size])
            # X_image = np.array(X_image)
            X = np.array(X)
            Y = np.array(Y)
            Y_future = np.array(Y_future)
            P = np.array(P)
        else:
            X = _X
            # X_image = _X_image
            Y = _Y
            Y_future = np.array([])
            P = np.array([])
        X = torch.from_numpy(X).to(torch.float32).to(self.device) # shape [num_windows, window_size, features_dim]
        # X_image = torch.from_numpy(X_image).to(torch.float32).to(self.device) # shape [num_windows, window_size, features_dim]
        target_type = target_type = torch.float32 if self.onehot else torch.long
        Y = torch.from_numpy(Y).to(target_type).to(self.device)
        Y_future = torch.from_numpy(Y_future).to(target_type).to(self.device)
        P = torch.from_numpy(P).to(torch.float32).to(self.device)
        # return X, X_image, Y, Y_future, P
        return X, Y, Y_future, P

        
    def _load_data(self):
        X = []
        X_image = []
        Y = []
        self.samples_per_trial = []
        
        for kinematics_path, video_path in self.files_path:
            if os.path.isfile(kinematics_path) and kinematics_path.endswith('.csv'):
                kinematics_data = pd.read_csv(kinematics_path)

                feature_names = kinematics_data.columns.to_list()[:-1] if not self.feature_names_ else self.feature_names_
                kin_data = kinematics_data.loc[:, feature_names]
                kin_label = kinematics_data.iloc[:,-1] # last column is always taken to be the target class
                # image_features = np.load(video_path)

                # limit by the length of the smaller tensor, image or kin
                last_index =  len(kin_data)
                kin_data = kin_data.iloc[:last_index]
                kin_label = kin_label.iloc[:last_index]
                # image_features = image_features[:last_index]

                # drop the frames where the label does not exist
                drop_ind = kin_label[kin_label == '-'].index
                kin_data = kin_data.drop(index=drop_ind)
                kin_label = kin_label.drop(index=drop_ind)
                # image_features = np.delete(image_features, drop_ind.tolist(), axis=0)

                # X_image.append(image_features)
                X.append(kin_data.values)
                Y.append(kin_label.values)
                self.samples_per_trial.append(len(kin_data))

        self.max_len = max([d.shape[0] for d in X])
        print(self.samples_per_trial)
        
        # label encoding and transformation
        if not self.target_names:
            self.le.fit(np.concatenate(Y))
        else:
            self.le.fit(np.array(self.target_names))
        self.target_names = self.le.classes_
        print(self.target_names)
        Y = [self.le.transform(yi) for yi in Y]
        print(Y[0])

        # one-hot encoding
        if self.onehot:
            self.enc.fit(np.arange(len(self.target_names)).reshape(-1, 1))
            Y = [yi.reshape(len(yi), 1) for yi in Y]
            Y = [self.enc.transform(yi) for yi in Y]
            print(Y[0])
        
        
        # store data inside a single 2D numpy array
        # X_image = np.concatenate(X_image)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        X_image = []

        return X, X_image, Y, 
    
    def __len__(self):
        # this should return the size of the dataset
        # return self.Y.shape[0] - self.observation_window_size - self.prediction_window_size - 1
        return int((self.Y.shape[0] )//self.observation_window_size)
        
        
    
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        start_idx = idx * self.observation_window_size
        end_idx = (idx + 1) * self.observation_window_size
        
        # features = self.X[idx + 1 : idx + self.observation_window_size + 1]
        features = self.X[start_idx : end_idx]
        # image_features = self.X_image[idx + 1 : idx + self.observation_window_size + 1]
        target = self.Y[start_idx : end_idx] # one additional observation is given for recursive decoding in recognition task
        gesture_pred_target = self.Y[idx + self.observation_window_size + 1 : idx + self.observation_window_size + self.prediction_window_size + 1]
        traj_pred_target = self.X[idx + self.observation_window_size + 1 : idx + self.observation_window_size + self.prediction_window_size + 1]
        
        # return kinematic_features, image_features, target, gesture_pred_target, traj_pred_target
        return features, target, gesture_pred_target, traj_pred_target

    
    @staticmethod
    def collate_fn(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), target_type=torch.float32, cast: bool = True):
        X = []
        # X_image = []
        Y = []
        Future_Y = []
        P = []
        # for x, xi, y, yy, p in batch:
        for x, y, yy, p in batch:
            X.append(x)
            # X_image.append(xi)
            Y.append(y)
            Future_Y.append(yy)
            P.append(p)
        # X_image = np.array(X_image)
        X = np.array(X)
        Y = np.array(Y)
        Future_Y = np.array(Future_Y)
        P = np.array(P)

        if cast:
            # cast to torch tensor
            x_batch = torch.from_numpy(X)
            # xi_batch = torch.from_numpy(X_image)
            y_batch = torch.from_numpy(Y)
            yy_batch = torch.from_numpy(Future_Y)
            p_batch = torch.from_numpy(P)
            
            # cast to appropriate data type
            x_batch = x_batch.to(torch.float32)
            # xi_batch = xi_batch.to(torch.float32)
            y_batch = y_batch.to(target_type)
            yy_batch = yy_batch.to(target_type)
            p_batch = p_batch.to(torch.float32)

            # cast to appropriate device
            x_batch = x_batch.to(device)
            # xi_batch = xi_batch.to(device)
            y_batch = y_batch.to(device)
            yy_batch = yy_batch.to(device)
            p_batch = p_batch.to(device)
        else:
            x_batch = X
            # xi_batch = X_image
            y_batch = Y
            yy_batch = Future_Y
            p_batch = P

        # return (x_batch, xi_batch, y_batch, yy_batch, p_batch)
        return (x_batch, y_batch, yy_batch, p_batch)


                