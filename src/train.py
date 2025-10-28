import argparse, json, joblib, yaml, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

USE_MLFLOW = False
try:
    import mlflow
    # Solo usar MLflow si existen las env vars (para DagsHub)
    if os.getenv("MLFLOW_TRACKING_URI"):
        USE_MLFLOW = True
except Exception:
    USE_MLFLOW = False

def load_params(pfile):
    with open(pfile) as f:
        return yaml.safe_load(f)

def train(params, data_path, model_path, metrics_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y
    )

    if params["model"]["type"] == "LogisticRegression":
        model = LogisticRegression(max_iter=params["model"]["max_iter"], C=params["model"]["C"])
    elif params["model"]["type"] == "SVC":
        model = SVC(C=params["model"]["C"], probability=True)
    else:
        raise ValueError("model.type no soportado")

    if USE_MLFLOW:
        mlflow.set_experiment("iris")
        with mlflow.start_run():
            mlflow.log_params(params["model"])
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            acc = accuracy_score(yte, pred)
            mlflow.log_metric("accuracy", acc)
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
    else:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)
    print(f"accuracy={acc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    p = load_params(a.params)
    train(p, a.data, a.model, a.metrics)