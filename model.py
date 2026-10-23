"""Load the saved model and provide image preprocessing and prediction."""

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Convert an image into a normalized RGB batch for the model."""
    with Image.open(img_path) as image:
        resized_image = image.convert("RGB").resize((224, 224))
        image_array = img_to_array(resized_image) / 255.0

    return image_array.reshape(1, 224, 224, 3)


def predict_result(predict):
    """Return the digit index with the highest prediction score."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
