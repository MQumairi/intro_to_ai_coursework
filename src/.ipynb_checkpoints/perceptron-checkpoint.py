from sklearn import metrics
import os
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Convert all missing values in the specified column to the median
def missing_median(dataset, name):
    med = dataset[name].median()
    df[name] = df[name].fillna(med)

path = "/Users/kamal/Documents/CS/Year3/ai/coursework/intro_to_ai_coursework/src"
    
filename_read = os.path.join(path, "dataset.csv")
dataset = pd.read_csv(filename_read, na_values=['NA', '?'])
missing_median(dataset, '')

# Assign X and y
X = dataset.drop('AveragePrice', axis=1)
y = dataset['AveragePrice']

# Splits the data into training data and validation data
kf = KFold(5)

sc = StandardScaler()

# Create a MLP, with its training parameters
mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 64),
                   max_iter=2000, activation="relu", random_state=1)

# using 5-fold cross validation, for each fold, a DT
# is trained and tested and confusion matrices generated
for train_index, validate_index in kf.split(X, y):
    Xt = X[train_index]
    sc.fit(Xt)
    X_train_std = sc.transform(Xt)
    X_test_std = sc.transform(X[validate_index])
    # Train the MLP with the training data
    mlp.fit(X_train_std, y[train_index])
    y_test = y[validate_index]
    # Use the validation data to make predictions with the trained DT
    y_pred = mlp.predict(X_test_std)
    print(y_test)
    print(y_pred)
    print(
        f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1
