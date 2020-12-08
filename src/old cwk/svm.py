from sklearn import metrics
import os
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

path = "/Users/kamal/Documents/CS/Year3/ai/coursework/intro_to_ai_coursework/src"
    
filename_read = os.path.join(path, "dataset.csv")
dataset = pd.read_csv(filename_read, na_values=['NA', '?'])

# Convert all missing values in the specified column to the median
def missing_median(dataset, name):
    med = dataset[name].median()
    dataset[name] = dataset[name].fillna(med)

#missing_median(dataset, '')

# Assign X and y
X = dataset.drop('AveragePrice', axis=1)
y = dataset['AveragePrice']

# perform a single split, with 25% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

sc = StandardScaler()

# Use the validation data to make predictions with the trained SVM
svm_model = SVR(kernel='linear', C=100,).fit(X, y)

# predict values for the testing data
y_pred = svm_model.predict(X_test)

# print accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

