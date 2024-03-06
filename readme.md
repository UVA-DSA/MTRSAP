# MTRSAP - Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction

This repository provides the code for the ICRA 2024 paper "Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction".

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

2. **Preprocessed Dataset**: Obtain the preprocessed dataset required for your project. Refer to the Usage section for detailed instructions on acquiring and incorporating the dataset.

3. **Operating System**: While the project is designed to be compatible with various operating systems, Ubuntu is the preferred environment.


### Installation

1. Create the conda environment using the environment file. ``` conda env create -f environment.yml```
2. Verify PyTorch was installed correclty.
3. Place the preprocessed data in the **ProcessedData**.
4. Verify the configuration is as required in ```config.py```. Learning parameters are defined in ```config.py```.

## Usage


To reproduce gesture recognition results use the following command with the original configuration.

*Note that the dataset required is not publicly available. Hence, please reach out to the original authors to obtain the data used for this work.*
1. [JIGSAWS Dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
2. [TCN Features](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master)

The model parameters and dataloader scripts needs to be changed to suit custom datasets. The current dataloader and config is designed for the above dataset. 

``` python train_recognition.py --model transformer --dataloader v1 --modality 16 ```

Results will be in the **results** folder specifically in following files.
1. ```train_results.json``` : Detailed results for each subject in LOUO setup.
2. ```Train_{task}_{model}_{date-time}.csv ``` : Final results of the run.


## Contributing

Please feel free to improve the model, add features and use this for research purposes.

If you have any questions, please feel free to reach out using the following email addresses (cjh9fw@virginia.edu, ydq9ag@virginia.edu)
## License

The code for this project is made available to the public via the  [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

Special Thanks to [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master) for providing features for the dataset and inspiring further development in action segmentation. 

