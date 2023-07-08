import cv2
import numpy as np
from keras.models import load_model

class Model:

# ==========
# Load Model
# ==========

    def __init__(self):
        try:
            self.cnn = load_model('./model_savestate.h5')
        except Exception as e:
            print('Error loading model: ', e)

# =================================
# Mold and Prediction on Test Image
# =================================

    def predict(self, image_path: str):
        px = 125
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (px, px))
        prediction = self.cnn.predict(img.reshape(1, px, px, 1), 
                                      verbose=0)
        return np.argmax(prediction) 