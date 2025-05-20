import os

# --- Forcer l'utilisation du CPU uniquement ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Réduire les logs TensorFlow

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse
from PIL import Image
import io
import gdown

app = FastAPI(title="API de Classification de Poisson")

# --- Variables globales ---
MODEL_PATH = "/tmp/model.h5"
DRIVE_ID = "1JtGnwRwNeEKpHqrbaHAt3Vdf4-qF6Qh5"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
model = None  # Le modèle sera chargé ici

# --- Limiter la taille des fichiers uploadés (5 Mo max) ---
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.headers.get("content-length"):
        size = int(request.headers["content-length"])
        if size > 5 * 1024 * 1024:  # 5MB
            return JSONResponse(content={"error": "Image trop grande. Max 5 Mo autorisés."}, status_code=413)
    return await call_next(request)

# --- Fonction pour télécharger le modèle ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle...")
        cache_dir = os.environ.get("GDOWN_CACHE_DIR", "/tmp/.cache/gdown")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["GDOWN_CACHE_DIR"] = cache_dir
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False, use_cookies=False)
    else:
        print("Modèle déjà téléchargé.")

# --- Chargement du modèle au démarrage ---
@app.on_event("startup")
def startup_event():
    global model
    download_model()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modèle chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        model = None

# --- Classes des poissons ---
class_names = ['bar_loup', 'calamar', 'crevette', 'maquereau', 'pageot_royale', 'pouple', 'sardine']

# --- Prédiction ---
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confiance = predictions[0][predicted_class] * 100

    return class_names[predicted_class], float(np.round(confiance, 2))

# --- Endpoint de prédiction ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")

    image_data = await file.read()

    try:
        prediction, confiance = predict_image(image_data)
        return JSONResponse(content={"class": prediction, "confiance": confiance})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur pendant la prédiction : {e}")
@app.get("/")
async def welcome():
    return JSONResponse(content={"welcome"})
