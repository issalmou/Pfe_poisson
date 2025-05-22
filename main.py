import os
import requests
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "/tmp/model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle depuis Dropbox...")
        url = "https://www.dl.dropboxusercontent.com/s/xxxxx/model_500mb.h5?dl=1"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Modèle téléchargé.")
        else:
            raise Exception(f"Erreur téléchargement : {response.status_code}")
    else:
        print("Modèle déjà présent.")

model = None

@app.on_event("startup")
def startup_event():
    global model
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé.")

class_names = ['bar_loup', 'calamar', 'crevette', 'maquereau', 'pageot_royale', 'pouple', 'sardine']

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100
    return class_names[predicted_class], float(round(confidence, 2))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    prediction, confidence = predict_image(image_data)
    return JSONResponse(content={"class": prediction, "confidence": confidence})
