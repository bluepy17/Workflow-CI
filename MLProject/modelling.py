import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
os.environ['MPLBACKEND'] = 'Agg'

def load_data():
    base_path = os.path.join(os.path.dirname(__file__), "heart_preprocessing")
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
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.25).astype(int)
    
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    mlflow.log_param("rf_n_estimators", 300)
    mlflow.log_param("rf_max_depth", 20)
    mlflow.log_param("rf_min_samples_leaf", 2)
    mlflow.log_param("rf_class_weight", "balanced")
    mlflow.log_param("gb_n_estimators", 200)
    mlflow.log_param("gb_learning_rate", 0.08)
    mlflow.log_param("gb_max_depth", 4)
    mlflow.log_param("gb_subsample", 0.8)
    mlflow.log_param("voting_weights", "2:1")
    mlflow.log_param("threshold", 0.25)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Heart_Disease_Voting_Classifier"
    )
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    train_model(X_train, X_test, y_train, y_test)
    print("Training finished")

if __name__ == "__main__":
    main()
