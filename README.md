# IN3062: Introduction to Artificial Intelligence Coursework
## Research Objective: "Predict police action following a stop and search".

Coursework for IN3062, by Kamal Kirosingh (KamalKirosingh), and Mohammed Alqumairi (MQumairi). 

Notice: in addition to the libraries covered in class, [Imbalance-learn](https://pypi.org/project/imbalanced-learn/) library is used for over sampling and under sampling. You'll need to "pip install imbalanced-learn" to run most notebooks.

All code is contained in the src folder, which in turn contains several items. Including:

## data.csv
The dataset, acquired from Kaggle (https://www.kaggle.com/sohier/london-police-records?select=london-stop-and-search.csv). Some columns irrelavant to the research question were removed.

## Cleaning Data.ipynb
#### By Mohammed Alqumairi
Contains the process we took for cleaning the dataset. 

## PCA.ipynb
#### By Kamal Kirosingh
Using Principal Component Analysis to reduce the number of dimensions to two, and plot the dataset on a 2D scatter graph. 

## Decision Tree.ipynb
#### By Kamal Kirosingh
An attempt to solve the research problem by using various Decision Tree algorithms.

## K Nearest Neighbor.ipynb
#### By Mohammed Alqumairi
An attempt to solve the research problem by using the K Nearest Neighbor algorithm.

## MLP.ipynb
#### By Mohammed Alqumairi
An attempt to solve the research problem by using Keras Neural Networks.

## models (folder)
#### By Mohammed Alqumairi
This contains:
- "Models tested.pdf": a tabulation of all neural network architectures tested, mapping them to a confusion matrix
- various .h5 files: saved neural network parameters, to not need to restart training from scratch every time
- "Confusion Matrix" folder: containing images of confusion matrices. Each image is identified with an interger, which "Models tested.pdf" maps to a confusion matrix (see the "Matrix Id" column in "Models tested.pdf")

## Perceptron.ipynb
#### By Kamal Kirosingh
An attempt to solve the research problem by using Perceptrons.

## SVM.ipynb
#### By Kamal Kirosingh
An attempt to solve the research problem by using Support Vector Machines.

## util.py & util_smote.py
#### By Mohammed Alqumairi
Contains various helper functions that are used by other files. Including functions for cleaning, preprocessing, plotting confusion matrices, and performing K Fold cross validations. 
util_smote.py in particular, has methods that are relevant to over sampling and under sampling the dataset.

## old cwk (folder)
A folder containing the work our team did for the previous research question on a different dataset. This was later abandoned because the dataset was too easy, which was not conducive to scoring well for this coursework. 
