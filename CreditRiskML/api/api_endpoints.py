from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
app = FastAPI()

with open(r"C:\Users\merty\Desktop\CreditRiskML\api\xgboost_tuned.pkl", "rb") as file:
    pipeline = pickle.load(file)

class CreditRiskML(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")



@app.post("/predict")
async def predict(features: CreditRiskML):
    print(features.model_dump())
    input_data = pd.DataFrame([features.model_dump()])
    print(input_data)

    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[0][1]

    return {"prediction": int(prediction[0]),
            "probability": float(probability)}