import numpy as np
import matplotlib.pyplot as plt
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# A function that takes in the target column, and prints the frequencies for each class.
# If you set the plot parameter to "True", the function will also graph the frequencies for a pictorial veiw of any imbalance


def display_frequencies(target=None, plot=True):
    labels, frequencies = np.unique(target, return_counts=True)

    for i in range(len(labels)):
        percentage_of_class = (frequencies[i] / np.sum(frequencies)) * 100
        percentage_of_class = np.around(percentage_of_class, 2)
        print(
            f"Class {labels[i]}: {percentage_of_class}%     ({frequencies[i]})")

    print(f"Total: {np.sum(frequencies)}")
    if(plot):
        plt.figure(figsize=(10, 8))
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.title("Frequency of Target Classes in Dataset")
        plt.bar(labels, frequencies)
        plt.show


# A function that takes in your X_train and y_train, and outputs over-sampled versions of the training set.
# You can specify how much each class can be over-sampled by passing in a sampling_strategy.
# Your sampling strategy will be a dictionary mapping the class (0, 1, 2, 3, 4, 5, 6, 7) to the frequency you wish for this class in the dataset.
def smote_data(X, y, sampling_strategy=None):
    if(sampling_strategy is None):
        oversample = SMOTE()
    else:
        oversample = SMOTE(sampling_strategy=sampling_strategy)
    X_smoted, y_smoted = oversample.fit_resample(X, y)
    return X_smoted, y_smoted
