from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import.
    """
    if not ARTIFACT_BUCKET:
        print("[WARN] ARTIFACT_BUCKET chua duoc dat. Bo qua download_model.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # 1. Tao storage.Client()
    client = storage.Client()

    # 2. Lay bucket va blob tuong ung
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)

    # 3. Tai file model xuong may
    blob.download_to_filename(MODEL_PATH)

    # 4. In thong bao thanh cong
    print(f"Model da duoc tai xuong tu gs://{ARTIFACT_BUCKET}/{MODEL_KEY} ve {MODEL_PATH}.")


if os.path.exists(MODEL_PATH) or ARTIFACT_BUCKET:
    try:
        download_model()
        model = joblib.load(MODEL_PATH)
        print("Model da duoc load thanh cong vao bo nho.")
    except Exception as e:
        print(f"[WARN] Chua the load model tai khoi dong: {e}")
        model = None
else:
    model = None


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # 6. Kiem tra so luong dac trung
    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 10 features (adult income), but got {len(req.features)}",
        )

    # 7. Goi model.predict([req.features])
    pred = int(model.predict([req.features])[0])

    # 8. Tra ve dict chua prediction va label
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
