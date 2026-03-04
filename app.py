import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/")
def home():
    return "Naive Bayes API Running"

@app.route("/train", methods=["POST"])
def train_model():

    file = request.files['file']
    target_column = request.form['target']

    df = pd.read_csv(file)

    # Remove ID columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    X = X.fillna(X.mode().iloc[0])

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    return jsonify({
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    })

if __name__ == "__main__":
    app.run()
    