import sys
import os
print(os.curdir)
sys.path.append('./')

def test_generated_distribution():

    parent_dir = ".."
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)

    import numpy as np
    import pandas as pd
    from naivebayesclassifier import NaiveBayes

    file_path = "/data.csv"
    data = pd.read_csv(current_dir + file_path)

    # Extract specific columns
    air_temp = data['Air temperature [K]'].values
    pro_temp = data['Process temperature [K]'].values
    rot_speed = data['Rotational speed [rpm]'].values
    failure_data = data['Target'].values

    # Combine features into a single array
    features = np.column_stack((air_temp, rot_speed))
    target = failure_data

    # Split the data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the Naive Bayes classifier
    nb = NaiveBayes()
    nb.fit(X_train, y_train) # Training

    # Predict on the test set
    y_pred = nb.predict(X_test)

    # Evaluate the model
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    gnb_accuracy = accuracy_score(y_test, y_pred_gnb)
    print(f"GNB Accuracy: {gnb_accuracy:.4f}")
    print("\nGNB Classification Report:")
    report = classification_report(y_test, y_pred_gnb)
    print(report)

    assert accuracy == gnb_accuracy

