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
from sklearn.tree import DecisionTreeRegressor


# def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar(fraction=0.05)
#     tick_marks = np.arange(len(names))
#     plt.xticks(tick_marks, names, rotation=45)
#     plt.yticks(tick_marks, names)
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


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
tree = DecisionTreeRegressor()

# using 5-fold cross validation, for each fold, a DT
# is trained and tested and confusion matrices generated
for train_index, validate_index in kf.split(X, y):
    Xt = X[train_index]
    sc.fit(Xt)
    X_train_std = sc.transform(Xt)
    X_test_std = sc.transform(X[validate_index])
    # Train the DT with the training data
    tree.fit(X_train_std, y[train_index])
    y_test = y[validate_index]
    # Use the validation data to make predictions with the trained DT
    y_pred = tree.predict(X_test_std)
    print(y_test)
    print(y_pred)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(
    metrics.mean_squared_error(y_test, y_pred)))

# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# cm = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure()
# plot_confusion_matrix(cm, values, title='')
