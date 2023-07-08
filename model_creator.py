# =======
# Imports
# =======

import os
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping

# =======================
# Directory and varieties
# =======================

train_dir = './Rice_Image_Dataset/Train/'
varieties = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
px = 125

# ===============================
# Molding data for keras function
# ===============================

def mold_training_data():
    training_data = []
    for variety in varieties:
        var_path = os.path.join(train_dir, variety)
        var_val = varieties.index(variety)
        for img in os.listdir(var_path):
            if '.DS_Store' != img:
                img_array = cv2.imread(os.path.join(var_path, img), 
                                       cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (px, px)) #Halve image size 
                training_data.append([new_array, var_val])

    random.shuffle(training_data) # Random order for better fitting

    return training_data

# ======================
# Molding data for keras
# ======================

training_data = mold_training_data()
Matrix_data = np.array([data[0] for data in training_data])
Val_data = np.array([data[1] for data in training_data])

# ==========================================
# Setting CNN (Convolutional Neural Network)
# ==========================================

cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
               input_shape=(px, px, 1)))
cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(32, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

# ====================
# Setting Compiler CNN
# ====================

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# ============================
# Setting Early Stop Condition
# ============================

early_stop = EarlyStopping(monitor='val_loss', 
                           mode='min', 
                           verbose=2, 
                           patience=2) # Overfit

# ===========
# Train model
# ===========

cnn.fit(Matrix_data, 
        Val_data, 
        callbacks=early_stop, 
        epochs=6,
        validation_split=0.15)

# =====================
# Save and export model
# =====================

cnn.save('model_savestate.h5')