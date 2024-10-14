# MTRSAP: Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction

This repository contains the official code for the paper _"Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction"_ submitted to ICRA 2024.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a **Multimodal Transformer** designed for **real-time surgical activity recognition and prediction**. It integrates various data modalities such as robotic kinematics and video data, offering an efficient approach to gesture recognition and activity prediction in surgical settings.

## Getting Started

Follow these steps to set up the project in your local environment.

### Prerequisites

Ensure you have the following installed:

1. **Anaconda**: Download and install [Anaconda](https://www.anaconda.com/products/distribution).
2. **Preprocessed Dataset**: Obtain the preprocessed dataset in the following format:
   ```bash
   ProcessedDatasets/
   └── Task/
       └── Task_S0X_T0Y.csv
   ```
   Each CSV file should have columns such as for the modalities:

   ```
   MTML_position_x, MTML_position_y, MTML_position_z, MTML_rotation_0, ..., resnet_0, ..., seg_0, ..., label
   ```

   DemoData folder should include a sample csv for your reference.
   See the [dataset format](#dataset-format) for more details.

3. **Operating System**: While the project is cross-platform, Ubuntu is the recommended environment.

### Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone https://github.com/your-repo/MTRSAP.git
   cd MTRSAP
   ```

2. Create a Conda environment using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```

3. Verify that **PyTorch** and other dependencies were installed correctly.

4. Place the preprocessed dataset into the `ProcessedData` directory.

5. Adjust configuration parameters as needed by editing `config.py`, where you can define learning rates, batch sizes, and other training parameters.

## Usage

To run the model for gesture recognition with the default settings, use the following command:

```bash
python train_recognition.py --model transformer --dataloader v2 --modality 16
```

To run the complete suite of experiments for gesture recognition using different modalities.
```bash
bash run_experiment.sh
```


### Results

The results will be saved in the `results/` directory, including:
1. **`train_results.json`**: Detailed results for each subject in the Leave-One-User-Out (LOUO) setup.
2. **`Train_{task}_{model}_{datetime}.csv`**: Final results of the training run.

## Contributing

We encourage you to contribute to the project by improving the model, adding new features, or using it for further research. Feel free to submit pull requests or raise issues.

For any questions, please reach out to us at:
- cjh9fw@virginia.edu
- ydq9ag@virginia.edu

## License

This project is distributed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

Special thanks to [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks) for providing foundational work on action segmentation, which inspired this research.