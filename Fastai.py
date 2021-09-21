import torch
from transformers import *
from fastai.text.all import *
from blurr.data.all import *
from blurr.modeling.all import *
import wordninja
import uvicorn
import numpy as np 
import joblib 
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

inf_learn = load_learner('/Users/brl.314/Downloads/Train_Aug2_distillbert-multilingual.pkl')

def pred(x):
    result=inf_learn.blurr_predict([x])
    pred, b, prob = zip(*result)
    return pred[0][0]
    
def proba(x):
    result=inf_learn.blurr_predict([x])
    pred, b, prob = zip(*result)
    prob_spam=float(prob[0][0][0].numpy())
    prob_not_spam=float(prob[0][0][1].numpy())
    return float(prob[0][0].max().numpy())

app = FastAPI(title="India Spam Model Fastai API for Questions")

class spamrequest():
    text: str

class response():
    prediction: str

@app.get("/india_spam")
async def predict_spam(content: str):
    
    predict = pred(content)
    probability = proba(content)
    if predict=='0':
        return {"Prediction":'Spam',"Probability": probability}
    else:
        return {"Prediction":'Not Spam',"Probability": probability}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=5000, debug=True)
