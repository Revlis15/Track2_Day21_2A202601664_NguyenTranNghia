import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Bonus 5: Kiem tra ty le lop duong (Data drift check)
    positive_ratio = float((df_train["target"] == 1).mean())
    reference_ratio = 0.248
    if abs(positive_ratio - reference_ratio) > 0.05:
        print(f"[CANH BAO] Ty le lop duong trong tap huan luyen ({positive_ratio:.4f}) lech qua 5% so voi ty le tham chieu ({reference_ratio:.4f})!")
    else:
        print(f"[INFO] Ty le lop duong hop le: {positive_ratio:.4f} (tham chieu: {reference_ratio:.4f})")

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # 3. Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # 4. Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Du doan tren tap holdout va tinh chi so mac dinh (threshold = 0.5)
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Dieu chinh nguong quyet dinh (Decision Threshold Tuning)
        best_threshold = 0.5
        best_f1 = f1
        try:
            probs = model.predict_proba(X_eval)[:, 1]
            thresholds = np.arange(0.1, 0.95, 0.05)
            for th in thresholds:
                th_preds = (probs >= th).astype(int)
                th_f1 = f1_score(y_eval, th_preds, zero_division=0)
                if th_f1 > best_f1:
                    best_f1 = float(th_f1)
                    best_threshold = float(th)
            print(f"[BONUS 2] Nguong mac dinh 0.5 (F1={f1:.4f}) vs Nguong toi uu {best_threshold:.2f} (F1={best_f1:.4f})")
        except Exception as e:
            print(f"Khong the tinh best threshold: {e}")

        # Bonus 3: Tao bao cao Precision / Recall & Confusion Matrix
        os.makedirs("outputs", exist_ok=True)
        cm = confusion_matrix(y_eval, preds)
        clf_rep = classification_report(y_eval, preds, target_names=["<=50K (0)", ">50K (1)"])
        detail_report = f"""=== BAO CAO CHI TIET MODEL ===
Accuracy: {acc:.4f}
F1 Score (target=1): {f1:.4f}
Best Threshold: {best_threshold:.2f} (Best F1: {best_f1:.4f})
Positive Ratio (Train): {positive_ratio:.4f}

--- CONFUSION MATRIX ---
TN: {cm[0, 0]} | FP: {cm[0, 1]}
FN: {cm[1, 0]} | TP: {cm[1, 1]}

--- CLASSIFICATION REPORT ---
{clf_rep}
"""
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_report)

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_ratio", positive_ratio)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.sklearn.log_model(model, "model")

        # 7. In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # 8. Luu metrics ra file outputs/report.json
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "positive_class_ratio": positive_ratio,
        }
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 9. Luu mo hinh ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # 10. Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
