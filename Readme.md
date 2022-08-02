This is a Python 3 implementation of Jinbo Xu's code for his paper "Distance-based protein folding powered by deep learning", PNAS August 20, 2019 116 (34). Sharing as per Prof. Xu's request.

[Paper](https://www.pnas.org/content/116/34/16856)

[Original code](https://github.com/j3xugit/RaptorX-Contact)

[Data (registration required)](http://raptorx.uchicago.edu/download)

### Minor fixes
- Indentation errors.
- Indexing error in the function `utils.MidpointFeature()`.
- Replaced `opt` package by `argparse` for parsing arguments.
- Displaying inference progress.

## Data preparation
After registration with an academic email, follow the link provided to access datasets and trained models.

Assume the working directory is 'DL4DistancePrediction2'.

### Input data
Input for the 2D-Dilated Resnet model are pre-processed **contact features** for protein sequence, in .pkl format.

For example, '76CAMEO.2015.contactFeatures.pkl'.

Download the data to local folder, e.g., './data'

### Trained models
There are 2 pretrained 2D-Dilated Resnet models provided: 'RXContact-DeepMode11410.pkl' and 'RXContact-DeepModel10820.pkl'.

Download them to local folder, e.g., './models'


## Running inference
From the working folder, run the main script `run_distance_predictor.py` for inference:

```console
python run_distance_predictor.py -m modelfiles -p predfiles [-d save_folder] [-g ground_truth_folder]
```
Parameters:

**modelfiles**: Specify one or multiple model files in PKL format, separated by semicolon.

**predfiles**: Specify one or multiple input feature files to be predicted. File(s) in PKL format, separated by semicolon in case of multiple input files.

**save_folder** (optional): Specify where to save the result files.

**ground_truth_folder** (optional): Specify the ground truth folder containing all the native atom-level distance matrix. When this option is provided, contact prediction accuracy will be calculated.

Example:
```console
python run_distance_predictor.py -p data/76CAMEO.2015.contactFeatures.pkl -m models/RXContact-DeepMode11410.pkl -d result/76CAMEO.2015
 ```
