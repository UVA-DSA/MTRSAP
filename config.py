import torch
from datautils.datagen import img_features,kinematic_feature_names,colin_features, segmentation_features, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables


RECORD_RESULTS = True

tcn_model_params = {
    "class_num": 7,
    "decoder_params": {
        "input_size": 256,
        "kernel_size": 61,
        "layer_sizes": [
            96,
            64,
            # 64
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel",
        "transposed_conv": True
    },
    "encoder_params": {
        "input_size": 25,
        "kernel_size": 61,
        "layer_sizes": [
            64,
            256,
            # 128
        ],
        "layer_type": "TempConv",
        "norm_type": "Channel"
    },
    "fc_size": 32,
    "mid_lstm_params": {
        "hidden_size": 64,
        "input_size": 256,
        "layer_num": 1
    }
}


transformer_params = {
    "d_model": 64,
    "nhead": 1,
    "num_layers": 1,
    "hidden_dim": 128,
    "layer_dim": 4,
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 14,
        "kernel_size": 31,
        "out_channels": 64,
    },
    "decoder_params": {
        "in_channels": 64,
        "kernel_size": 31,
        "out_channels": 64
    },
}

learning_params = {
    # "lr": 8.906324028628413e-5,
    "lr": 1e-05,
    "epochs": 20,
    "weight_decay": 1e-4,
    "patience": 3
}

dataloader_params = {
    
    "batch_size": 10,
    "one_hot": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "observation_window": 30,
    "prediction_window": 10,
    "user_left_out": 7,
    "cast": True,
    "include_image_features": False,
    "normalizer": '',  # ('standardization', 'min-max', 'power', '')
    "step": 1,  # 1 - 30 Hz
    "context": 9  # 0-nocontext, 1-contextonly, 2-context+kin, 3-imageonly, 4-image+kin, 5-image+kin+context, 6-colin_features, 7- colin+context, 8-colin+kin, 9-colin+kin+context, 10-segonly, 11-kin+seg, 12-kin+seg+context, 13-kin+seg+context+colins, 14-seg+colins
    # hamid -  do not need (1,3,5,7)
    
}

# --context 
modality_mapping = {
    0: kinematic_feature_names_jigsaws[38:],  # Kinematic (38)
    1: kinematic_feature_names_jigsaws_patient_position,  # Kinematic (14)
    2: state_variables,  # Context (GT)
    3: colin_features,  # Colins Features
    4: img_features,  # ResNet50 Features
    5: segmentation_features,  # Segmentation Masks Features
    6: state_variables + colin_features,  # Context(GT) + Colins Features
    7: kinematic_feature_names_jigsaws[38:] + state_variables,  # Kinematic (38) + Context(GT)
    8: kinematic_feature_names_jigsaws_patient_position + state_variables,  # Kinematic (14) + Context(GT)
    9: kinematic_feature_names_jigsaws[38:] + colin_features,  # Kinematic (38) + Colins Features
    10: kinematic_feature_names_jigsaws_patient_position + colin_features,  # Kinematic (14) + Colins Features
    11: kinematic_feature_names_jigsaws[38:] + img_features,  # Kinematic (38) + ResNet50
    12: kinematic_feature_names_jigsaws_patient_position + img_features,  # Kinematic (14) + ResNet50
    13: kinematic_feature_names_jigsaws[38:] + segmentation_features,  # Kinematic (38) + Segmentation Masks
    14: kinematic_feature_names_jigsaws_patient_position + segmentation_features,  # Kinematic (14) + Segmentation Masks
    15: kinematic_feature_names_jigsaws[38:] + state_variables + colin_features,  # Kinematic (38) + Context(GT) + Colins Features
    16: kinematic_feature_names_jigsaws_patient_position + state_variables + colin_features,  # Kinematic (14) + Context(GT) + Colins Features
    17: kinematic_feature_names_jigsaws[38:] + segmentation_features + state_variables,  # Kinematic (38) + Segmentation Masks + Context(GT)
    18: kinematic_feature_names_jigsaws_patient_position + segmentation_features + state_variables,  # Kinematic (14) + Segmentation Masks + Context(GT)
    19: kinematic_feature_names_jigsaws[38:] + segmentation_features + state_variables,  # Kinematic (38) + Segmentation Masks + Context(GT) + Colins
    20: kinematic_feature_names_jigsaws_patient_position + segmentation_features + state_variables,  # Kinematic (14) + Segmentation Masks + Context(GT) + Colins
    21: kinematic_feature_names_jigsaws_patient_position  + img_features,  # Kinematic (14)  + Resnet
}


