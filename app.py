# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import io

# from keras_multi_head import MultiHeadAttention

# # Load model ONCE at startup
# model = tf.keras.models.load_model(
#     "hair-diseases.hdf5",
#     custom_objects={"MultiHeadAttention": MultiHeadAttention}
# )

# class_names = [
#     "Alopecia Areata",
#     "Contact Dermatitis",
#     "Folliculitis",
#     "Head Lice",
#     "Lichen Planus",
#     "Male Pattern Baldness",
#     "Psoriasis",
#     "Seborrheic Dermatitis",
#     "Telogen Effluvium",
#     "Tinea Capitis"
# ]

# app = FastAPI()

# # Allow Android access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     image = image.resize((128, 128))

#     img = np.array(image) / 255.0
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)[0]
#     idx = int(np.argmax(preds))

#     return {
#         "disease": class_names[idx],
#         "confidence": float(preds[idx])
#     }
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import gdown

from keras_multi_head import MultiHeadAttention

# =========================
# MODEL DOWNLOAD SECTION
# =========================

MODEL_PATH = "hair-diseases.hdf5"

# 🔴 REPLACE THIS WITH YOUR REAL FILE ID
MODEL_URL = "https://drive.google.com/uc?id=1Ypd-C6CQB7_Y3xuefVysi9GU4L0D_819"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)


# =========================
# LOAD MODEL (ONCE)
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"MultiHeadAttention": MultiHeadAttention}
)

print("Model loaded successfully!")

# =========================
# CLASS NAMES
# =========================

class_names = [
    "Alopecia Areata",
    "Contact Dermatitis",
    "Folliculitis",
    "Head Lice",
    "Lichen Planus",
    "Male Pattern Baldness",
    "Psoriasis",
    "Seborrheic Dermatitis",
    "Telogen Effluvium",
    "Tinea Capitis"
]

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((128, 128))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    return {
        "disease": class_names[idx],
        "confidence": float(preds[idx])
    }
