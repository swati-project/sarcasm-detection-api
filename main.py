from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sarcasm

app = FastAPI(title="Sarcasm Detection API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    result = predict_sarcasm(data.text)
    return {"sarcasm": bool(result)}
