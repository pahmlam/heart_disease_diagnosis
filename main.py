from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from utils import add_new_features


app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = 'model/heart_model.pkl'

# Tải mô hình
model_pipeline = None
if os.path.exists(model_path):
    try:
        model_pipeline = joblib.load(model_path)
        print("Đã tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {str(e)}")
else:
    print(f"CẢNH BÁO: Không tìm thấy file {model_path}. Vui lòng chạy train.py trước.")

class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: HeartData):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải. Vui lòng kiểm tra server logs.")

    try:
        input_data = pd.DataFrame([data.dict()])
        
        # Tính toán Feature Engineering cho hiển thị (Frontend)
        features_info = {}
        if data.age > 0:
            features_info["chol_per_age"] = round(data.chol / data.age, 3)
            features_info["bps_per_age"] = round(data.trestbps / data.age, 3)
            features_info["hr_ratio"] = round(data.thalach / data.age, 3)
        else:
            features_info = {"chol_per_age": 0, "bps_per_age": 0, "hr_ratio": 0}

        prediction = model_pipeline.predict(input_data)
        probability = model_pipeline.predict_proba(input_data)
        
        result_text = "Có nguy cơ bệnh tim" if prediction[0] == 1 else "Bình thường"
        prob_percent = probability[0][1] * 100 if prediction[0] == 1 else probability[0][0] * 100
        
        return {
            "prediction": int(prediction[0]),
            "result_text": result_text,
            "confidence": round(prob_percent, 2),
            "features_engineering": features_info
        }
    except Exception as e:
        # In lỗi ra console để debug
        print(f"Lỗi dự đoán: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)