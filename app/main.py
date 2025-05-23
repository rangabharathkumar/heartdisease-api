from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_utils import predict_heartdisease

app = FastAPI(title="Heart Disease Prediction API")

class InputData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalachh: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# ✅ This resolves the 404 issue for root "/"
@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API. Use POST /predict with input data."}

@app.post("/predict")
def get_prediction(data: InputData):
    try:
        prediction = predict_heartdisease(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
