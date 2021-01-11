# Project Title

This is the code repository for the 41st 2020 UCT final year undergraduate research project, whereby three machine learning models (Neural Network, Random Forest, Autoencoder) were built and evaluated on CFD data sourced from a simple 2D T-piece mixing domain. This repository contains all the preprocessing, training, and evaluation files for this project; along with the supporting source data, and original research report. All the necessary files in this project were compiled in python with the added capability of being implementable on a standard laptop/CPU device.


## Getting Started

According to the memory results available in the [main research report](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/Undergraduate_Report.pdf), the following approximate memory statistics are required to host a. Clone of the repository:

* *Volatile Memory* — *800MB*
* *Non-Volatile Memory* — *22GB*

Along with these dependancies in order for the program to compile in accordance with the observations recorded in the research report, the following libraries versions are required:

* *Python* — *3.7.4*
* *Matplotlib* — *3.3.1* 
* *NumPy* — *1.19.1* 
* *Pandas* — *1.14* 
* *PyTorch* — *1.7.0* 
* *Scikit-learn* — *0.23.1*

## Project Compilation

Before running the various code, the respective project parameters must be tuned (found in [parameters.py](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/Parameters.py)). This includes indicating the local directory of the repository folder, training status of the machine learning models, dataset evaluation range, machine learning model to be analysed, and the fluid property which is to be visualised during contour generation.

Once the project parameters have been modified, the project can be compiled by running the following steps:

# Step 1 
Load the CFD data by running [data_loader.py](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/data_loader.py).

This can be achieved by entering the following into the shell:
```
python ./data_loader.py
```

# Step 2 
Preprocess the CFD data by running [preprocess.py](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/preprocessing.py).

This can be achieved by entering the following into the shell:
```
python ./preprocess.py
```

# Step 3
If training mode has been initialised then the respective models can be trained by running [model_training.py](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/model_training.py).

This can be achieved by entering the following into the shell:
```
python ./model_training.py
```

Else the model performance can be evaluated by running [model_evaluation.py](https://github.com/Daniel-Ferrini/Undergraduate_Research_Project/blob/main/model_evaluation.py).

This can be achieved by entering the following into the shell:
```
python ./model_evaluation.py
```

Once each the python files have completely compiled the result of the evaluation should indicate the machine learning algorithm’s metadata, and the resulting CFD prediction contour and error frequency distributions as such:


()

## Acknowledgments
All aknowledgments for the project development goes to: 

* Supervisor -- Dr Ryno Laubscher

