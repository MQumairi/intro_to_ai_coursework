import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# A function that takes the data in, and cleans it

def data_cleaner(data):
    # Programatically remove the columns:
    #   - "Outcome linked to object of search"
    #   - "Removal of more than just outer clothing".
    del data["Outcome linked to object of search"]
    del data["Removal of more than just outer clothing"]

    # Convert latitude and longitude nulls to median
    lat_median = data["Latitude"].median()
    lon_median = data["Longitude"].median()

    data["Latitude"] = data["Latitude"].fillna(lat_median)
    data["Longitude"] = data["Longitude"].fillna(lon_median)

    # Change the "Date" column to type DateTime
    data['Date'] = pd.to_datetime(data['Date'])

    # Remove Oct-17 values in Age ranges
    # Reference for dictionary idea to replace values: https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers
    oct17_to_None = {"Oct-17": None}
    data = data.applymap(lambda s: oct17_to_None.get(s)
                         if s in oct17_to_None else s)

    # For the other columns, we'll need to drop the null values
    data = data.dropna()

    return data


# A function that takes the data in and encodes all non-numerals

def data_encoder(data):
    # Build a dictionary of Encoders
    encoders = {}

    # List the categorical cols
    categorical_cols = ["Type", "Date", "Gender", "Age range",
                        "Officer-defined ethnicity", "Legislation", "Object of search", "Outcome"]

    # Build an encoder for each categorical_col, and fit it to the values under that column
    for label in categorical_cols:
        encoders[f"{label} Encoder"] = LabelEncoder()
        encoders[f"{label} Encoder"].fit(data[label])

    # We copy the "data" variable into "data_encoded", such that changing one won't impact the other
    data_encoded = data.copy()

    # We perform the encoding to "data_encoded"
    for label in categorical_cols:
        data_encoded[label] = encoders[f"{label} Encoder"].fit(
            data[label]).transform(data[label])

    # Print the data_encoded... notice all values have been numerified!
    return (data_encoded, encoders)


# A function that plots a confusion matrix
# Retrieved from ex3Part2 (Lab 3) of the intro to AI module
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=20)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# A function that takes in y_test, model predictions, and a list of target classes, and plots a confusion matrix
police_actions_simple = ["Article Found", "Resolved", "Nothing",
                         "Cautioned", "Drug Warning", "Penalty Notice", "Arrested", "Summonsed"]


def confusion_plot(y_test, y_predictions, target_classes=police_actions_simple, title="Confusion Matrix", normalize=True):
    cm = confusion_matrix(y_test, y_predictions)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(16, 10))
    if(normalize):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized, target_classes,
                              title=title)
    else:
        plot_confusion_matrix(cm, target_classes,
                              title=title)
    plt.show()


# A function that takes in a model, and a dataframe, and trains the model using K-fold
def k_fold_train(model, data, folds=5):
    print("🚫 WARNING 🚫: this function will take time to process.")
    print(" ")
    # Set up a scaler
    sc = StandardScaler()

    # Use 5-fold split
    kf = KFold(folds)

    fold = 1
    for train_index, validate_index in kf.split(data):
        # Split into train and validate sets
        train_DF = pd.DataFrame(data.iloc[train_index, :])
        test_DF = pd.DataFrame(data.iloc[validate_index])

        # Split training set into ys and x's
        y_train_kfold = train_DF["Outcome"]
        x_train_kfold = train_DF.drop('Outcome', axis=1)

        # Split validation set into ys and x's
        y_test_kfold = train_DF["Outcome"]
        x_test_kfold = train_DF.drop('Outcome', axis=1)

        # Train model
        model.fit(x_train_kfold, y_train_kfold)

        # Predict
        print(
            f"Fold #{fold}, Training Size: {len(train_DF)}, Validation Size: {len(test_DF)}")
        print(
            f"Training Score: {model.score(x_train_kfold, y_train_kfold)}")
        print(f"Testig Score: {model.score(x_test_kfold, y_test_kfold)}")
        print("\n")
        fold += 1
