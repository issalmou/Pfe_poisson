from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image
import io
import gdown
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR only

import requests

app = FastAPI(title="API de Classification de Poisson")

MODEL_PATH = "/tmp/model.h5"
# DRIVE_ID = "1JtGnwRwNeEKpHqrbaHAt3Vdf4-qF6Qh5"
# DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

model = None  # variable globale pour le modèle chargé

# --- Gestion GPU/CPU propre ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # réduire logs TensorFlow

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print(f"GPU(s) détecté(s) : {[gpu.name for gpu in gpus]}")
#     except RuntimeError as e:
#         print(f"Erreur configuration GPU : {e}")
# else:
#     print("Aucun GPU détecté, utilisation du CPU.")

# --- Fonction téléchargement modèle ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle depuis Dropbox...")
        url = "https://www.dl.dropboxusercontent.com/scl/fi/XXXXX/vgg16_model_v2_20250518_200921.h5?...&dl=1"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Modèle téléchargé avec succès.")
        else:
            raise Exception(f"Erreur de téléchargement : {response.status_code}")
    else:
        print("Modèle déjà présent.")

# --- Chargement du modèle au démarrage ---
@app.on_event("startup")
def startup_event():
    global model
    download_model()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        model = None

# Définir les noms des classes dans le bon ordre
class_names = ['bar_loup', 'calamar', 'crevette', 'maquereau', 'pageot_royale', 'pouple', 'sardine']

def predict_image(image_bytes):
    # Convertir les données binaires en image PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Redimensionner à la taille attendue par VGG16 (224x224)
    image = image.resize((224, 224))

    # Convertir en tableau numpy + normaliser
    image_array = np.array(image) / 255.0

    # Ajouter une dimension batch (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    # Prédire
    predictions = model.predict(image_array)
    
    # Récupérer l'index de la classe prédite
    predicted_class = np.argmax(predictions, axis=1)[0]
    confiance = predictions[0][predicted_class] * 100
    # Retourner le nom de la classe
    return class_names[predicted_class],float(np.round(confiance,2))

@app.get("/")
def root():
    return {"message": "API is working"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire le contenu du fichier image
    image_data = await file.read()

    # Appeler la fonction de prédiction
    prediction,confiance = predict_image(image_data)
    
    return JSONResponse(content={"class": prediction,'confiance':confiance})
