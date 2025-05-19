import numpy as np
from tensorflow.keras.models import load_model as _load_model
from image_preprocessor import preprocess_image

CLASS_NAMES = { 0: 'no_siren', 1: 'yes_siren'}

def load_trained_model(model_path):
    return _load_model(model_path)

def predict_image(model, img_path):

    img_final = preprocess_image(img_path)
    predictions = model.predict(img_final)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class])
    return CLASS_NAMES[predicted_class], confidence
