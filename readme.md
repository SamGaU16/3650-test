# Test accuracy: 95.39%
## Classification Exam FISI-3650 - Universidad de los Andes
### Samuel Mateo Montañez Gil 202014559

Tensorflow Keras was the library used to training the model.
OpenCV (cv2) was used for reading the images.
The dataset was mold to halve the pixels.
The model:
- Uses two convolution 2D layers, each have a 'Max Pooling' operation.
- Uses a Flattens and two Dense layers, the last one for normalize the output.
- Has an early stop for overfitting.
- Has a validation set made up of about 15% of the training data.
- Was exported as 'model_savestate.h5'.
