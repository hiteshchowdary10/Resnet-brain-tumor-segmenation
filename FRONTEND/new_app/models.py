from django.db import models
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load model only once when server starts
MODEL_PATH = r"C:\Users\guntu\Music\Brain Tumour Resnet\FRONTEND\brain_tumor_RESNET (1).h5"
model = load_model(MODEL_PATH, compile=False)

# Correct class order (MUST match LabelEncoder)
CLASS_NAMES = [
    'glioma_tumor',
    'meningioma_tumor',
    'no_tumor',
    'pituitary_tumor'
]

TARGET_SIZE = (124, 124)


def preprocess_image(img):
    """
    Preprocess uploaded image exactly like training.
    NO scaling.
    Resize to 124x124.
    """
    image = Image.open(img)
    image = image.convert('RGB')
    image = image.resize(TARGET_SIZE)

    img_array = np.array(image).astype(np.float32)

    # DO NOT divide by 255
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_tumor(img):
    """
    Predict tumor class from uploaded image.
    """
    processed_img = preprocess_image(img)

    predictions = model.predict(processed_img)

    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions) * 100)

    return {
        "label": CLASS_NAMES[class_index],
        "confidence": confidence,
        "class_index": int(class_index),
        "raw_probabilities": predictions.tolist()
    }