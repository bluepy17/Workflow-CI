import os
import json
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))


def load_data():
    base_path = "heart_preprocessing"

    X_train = pd.read_csv(os.path.join(base_path, "X_train_preprocessing.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test_preprocessing.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train_preprocessing.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[2, 1],
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.25).astype(int)

    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "threshold": 0.25
    }

    with open("metric_info.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact("metric_info.json")
    os.remove("metric_info.json")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("training_confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("training_confusion_matrix.png")
    os.remove("training_confusion_matrix.png")


def main():
    X_train, X_test, y_train, y_test = load_data()
    train_model(X_train, X_test, y_train, y_test)
    print("Training finished")


if __name__ == "__main__":
    main()
