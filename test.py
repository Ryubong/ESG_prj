from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def social_process_data_and_predict(input_data, datafile, features, target):
    # load file
    df = pd.read_csv(datafile, encoding='cp949')

    # data preprocessing
    test = df[df.isna()[target]]
    train = df[df.notnull()[target]]

    # Remove class 'D'
    train = train[train[target] != 'D']

    le = LabelEncoder()
    train[target] = le.fit_transform(train[target])

    X_train = train[features]
    y_train = train[target]

    # ADASYN object
    adasyn = ADASYN(random_state=42)
    sampling_strategy = {0: 1700, 1: 1700, 2: 1200, 3: 1200, 4: 500}
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    
    # Create Gradient Boosting Classifier with hyperparameters
    gb_clf = GradientBoostingClassifier(n_estimators=230, learning_rate=0.2, max_depth=6, min_samples_leaf=30, ccp_alpha=0.000051, random_state=42)

    # Train the model
    gb_clf.fit(X_train_resampled, y_train_resampled)

    # Deriving forecasts
    predicted_grade = gb_clf.predict(input_data)

    # Decode the predicted environment class into the actual environment class
    predicted_grade_label = le.inverse_transform(predicted_grade)
    return predicted_grade_label

def scenario_analysis(model, encoder, data, features, scenario):
    # Adjust the data according to the new scenario
    modified_data = np.array(data)  # Copy the old data.
    for feature in features:
        # Find the index of the attribute.
        index = features.index(feature)
        # Change the value of that attribute.
        modified_data[0][index] += scenario.get(feature, 0)

    # Predict environment class with modified data
    predicted_grade = model.predict(modified_data)

    # Decode the predicted environment class into the actual environment class
    predicted_grade_label = encoder.inverse_transform(predicted_grade)

    return predicted_grade_label[0]
