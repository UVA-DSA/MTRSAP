# MTRSAP - Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction

This repository provides the code for the ICRA 2024 paper ["Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction"](https://arxiv.org/pdf/2403.06705).

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Introduction

This repo is the official code for the ICRA 2024 paper "Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction"

## Getting Started

Please follow the below instructions to setup the code in your environment.

### Prerequisites

1. **Anaconda**: Make sure to have Anaconda installed on your system. You can download it from Anaconda's official website.

2. **Operating System**: While the project is designed to be compatible with various operating systems, Ubuntu is the preferred environment.


### Installation

1. Create the conda environment using the environment file. ``` conda create -n mtrsap python=3.9 -y```
2. Activate the newly created environment ```conda activate mtrsap```
3. Install required python packages ```pip install -r requirements.txt```
4. Verify PyTorch was installed correclty. [Install Torch](https://pytorch.org/get-started/locally/)
5. Verify the configuration is as required in ```config.py```. Learning parameters are defined in ```config.py```.

## Usage

### Obtain the Required Data

1. The experiments are performed over the [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/). The original dataset does not contain the transcirptions for "surgical state variables". To run the experiments, please download the [COMPASS dataset](https://github.com/UVA-DSA/COMPASS/tree/main) which includes JIGSAWS data with additional annotations, and place the `Datasets` forlder within this repository.

2. The spatio-temporal features extracted from video files should be obtained from the authors, [Spatial Features](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master). After obtaining the spatial features from the original authors, please place them inside this repository. Make sure the folder containing the data is named `SpatialCNN`. The train/test splits specifications can also be obtained from [this git repository](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master/splits). Download the splits folder, and place it inside the `SpatialCNN` folder, next to the data folder.

3. To run the data preprocessing scripts, recognition and prediction pipelines you also need the video features extracted by a ResNet50 backbone, and instrument segmentation masks. Please contact us to obtain these features (cjh9fw@virginia.edu, ydq9ag@virginia.edu). After obtaining these, place them inside this repository with the original folder names.

In summary, all these data folders need to be present inside this repository to proceed with running pipelines:
  ```bash
   Datasets/
   └── dV/
   SpatialCNN/
   └── data/
       splits/
   segmentation_masks/
   └── outputs/
       pca_features/
       pca_features_normalized/
   resnet_features
   └── Knot_Typing/
       Needle_Passing/
       Suturing/
       Peg_Transfer/
   ```

### Preprocessing the data

1. Run the preprocessing script to generate the processed dataset. Replace `{task}` with the desired task, from the set of available tasks ("Peg_Transfer", "Suturing", "Knot_Tying", "Needle_Passing").
```bash
python data/datagen.py {task}
```
The preprocessed data should be generated in the following format, where `Task` is the same as the one you specified when running the `datagen` script:
   ```bash
   ProcessedDatasets/
   └── Task/
       └── Task_S0X_T0Y.csv
   ```
   Each CSV have the following columns:

   ```
   PSML_poaition_x, ..., PSMR_position_x, ..., left_holding, ..., right_holding, ..., label
   ```  

DemoData folder should include a sample csv for your reference.


### Run the Recognition Pipeline

To run the model for gesture recognition with the default settings, use the following command:

```bash
python train_recognition.py --model transformer --dataloader v2 --modality 16
```

To run the complete suite of experiments for gesture recognition using different modalities.
```bash
bash run_experiment.sh
```

Results will be in the **results** folder specifically in following files.
1. ```train_results.json``` : Detailed results for each subject in LOUO setup.
2. ```Train_{task}_{model}_{date-time}.csv ``` : Final results of the run.


### Run the Prediction Pipeline

To run the model for gesture prediction with the default settings, use the following command:

```bash
python train_prediction.py
```


## Contributing

Please feel free to improve the model, add features and use this for research purposes.

If you have any questions, please feel free to reach out using the following email addresses (cjh9fw@virginia.edu, ydq9ag@virginia.edu)
## License

The code for this project is made available to the public via the  [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

Special Thanks to [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master) for providing features for the dataset and inspiring further development in action segmentation. 

## Citation
If you find this dataset, model, or any of the features helpful in your research, please cite our paper. Proper citation helps the community and allows us to continue providing these resources.

You can cite the paper using the following BibTeX entry:

```bibtex
@INPROCEEDINGS{10611048,
  author={Weerasinghe, Keshara and Roodabeh, Seyed Hamid Reza and Hutchinson, Kay and Alemzadeh, Homa},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Multimodal Transformers for Real-Time Surgical Activity Prediction}, 
  year={2024},
  pages={13323-13330},
  keywords={Computational modeling;Computer architecture;Kinematics;Streaming media;Predictive models;Transformers;Real-time systems},
  doi={10.1109/ICRA57147.2024.10611048}}
```
Thank you for your support!



