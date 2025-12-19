import os
import json
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# DATA LOADING

def load_data():
    base_path = "heart_preprocessing"

    X_train = pd.read_csv(os.path.join(base_path, "X_train_preprocessing.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test_preprocessing.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train_preprocessing.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test

# MODEL TRAINING 

def train_model(X_train, X_test, y_train, y_test):

    # Ambil run_id dari MLflow Project
    run_id = os.environ.get("MLFLOW_RUN_ID")

    with mlflow.start_run(run_id=run_id):

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )

        model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb)],
            voting="soft",
            weights=[2, 1],
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.25).astype(int)

        print(classification_report(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        mlflow.log_param("threshold", 0.25)

        metric_info = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "threshold": 0.25,
        }

        with open("metric_info.json", "w") as f:
            json.dump(metric_info, f, indent=4)

        mlflow.log_artifact("metric_info.json")
        os.remove("metric_info.json")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_path = "training_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        return model

# MAIN

def main():
    X_train, X_test, y_train, y_test = load_data()
    train_model(X_train, X_test, y_train, y_test)
    print("Training finished successfully")


if __name__ == "__main__":
    main()
