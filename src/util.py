import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import metrics


# A function that takes the data in, and cleans it

def data_cleaner(data, numerify_dates=False):
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

    if numerify_dates:
        data['Date'] = pd.to_numeric(data['Date'])

    # Remove Oct-17 values in Age ranges
    # Reference for dictionary idea to replace values: https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers
    oct17_to_None = {"Oct-17": None}
    data = data.applymap(lambda s: oct17_to_None.get(s)
                         if s in oct17_to_None else s)

    # For the other columns, we'll need to drop the null values
    data = data.dropna()

    return data


# A function that takes the data in and encodes all non-numerals

def data_encoder(data, encode_dates=True):
    # Build a dictionary of Encoders
    encoders = {}

    # List the categorical cols
    categorical_cols = ["Type", "Gender", "Age range",
                        "Officer-defined ethnicity", "Legislation", "Object of search", "Outcome"]

    if encode_dates:
        categorical_cols.append("Date")

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
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues, rotation=90):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=rotation)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# A function that takes in y_test, model predictions, and a list of target classes, and plots a confusion matrix
police_actions_simple = ["Article Found", "Resolved", "Nothing",
                         "Cautioned", "Drug Warning", "Penalty Notice", "Arrested", "Summonsed"]


def confusion_plot(y_test, y_predictions, target_classes=police_actions_simple, title="Confusion Matrix", fontsize=18, normalize=True, rotation=90):
    cm = confusion_matrix(y_test, y_predictions)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(16, 10))
    plt.rcParams.update({'font.size': fontsize})
    if(normalize):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized, target_classes,
                              title=title, rotation=rotation)
    else:
        plot_confusion_matrix(cm, target_classes,
                              title=title, rotation=rotation)
    plt.show()


# A function that takes in a model, and a dataframe, and trains the model using K-fold
def k_fold_train(model, data, folds=5, is_NN=False, verbose=0, epochs=128):
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

        # Fit
        model.fit(x_train_kfold, y_train_kfold)

        # Evaluate
        y_predict_train = model.predict(x_train_kfold)
        y_predict_test = model.predict(x_test_kfold)
        print(
            f"Fold #{fold}, Training Size: {len(train_DF)}, Validation Size: {len(test_DF)}")

        # Accuracy
        print(f"Accuracy (Train): {model.score(x_train_kfold, y_train_kfold)}")
        print(f"Accuracy (Test): {model.score(x_test_kfold, y_test_kfold)}")

        # Precision
        precision_train = metrics.precision_score(
            y_train_kfold, y_predict_train, average="weighted")
        precision_test = metrics.precision_score(
            y_test_kfold, y_predict_test, average="weighted")
        print(f"Precision (Train): {precision_train}")
        print(f"Precision (Test): {precision_test}")

        # MCC
        mcc_train = metrics.matthews_corrcoef(y_train_kfold, y_predict_train)
        mcc_test = metrics.matthews_corrcoef(y_test_kfold, y_predict_test)
        print(f"MCC (Train): {mcc_train}")
        print(f"MCC (Test): {mcc_test}\n")

        fold += 1


# A function that takes the encoded data in, and returns the data "binarified"
def binarify_from_raw(data, nothing_value=0):
    # Replace all values in Outcome column, with 0 if "Nothing found - no further action", else 1.
    # Reference for dictionary idea to replace values: https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers
    outcome_splitter = {
        "Article found - Detailed outcome unavailable": 1,
        "Local resolution": 1,
        "Nothing found - no further action": nothing_value,
        "Offender cautioned": 1,
        "Offender given drugs possession warning": 1,
        "Offender given penalty notice": 1,
        "Suspect arrested": 1,
        "Suspect summonsed to court": 1
    }
    data_binary = data.applymap(lambda s: outcome_splitter.get(
        s) if s in outcome_splitter else s)
    return data_binary


# A function that takes the encoded data in, and returns the data "binarified"
def binarify_from_encoded(data, nothing_value=0):
    # Reference for dictionary idea to replace values: https://stackoverflow.com/questions/17114904/python-pandas-replacing-strings-in-dataframe-with-numbers
    outcome_splitter = {
        0: 1,
        1: 1,
        2: nothing_value,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1
    }
    data_binary = data.applymap(lambda s: outcome_splitter.get(
        s) if s in outcome_splitter else s)
    return data_binary


def pie_chart_y(y, title="Ratio of Classes"):
    labels, frequencies = np.unique(y, return_counts=True)
    plt.figure(figsize=[20, 5])
    fig1, ax1 = plt.subplots()
    ax1.pie(frequencies, labels=labels,
            autopct='%1.1f%%', shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    plt.title(title)
    plt.show()
