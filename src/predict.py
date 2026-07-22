import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

# Load the trained model
model = load_model("models/brain_tumor_model.keras")

# Enter the path of the MRI image you want to test
image_path = "dataset/test/yes/Te-glTr_0000.jpg"

# Read and preprocess the image
image = cv2.imread(image_path)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make prediction
prediction = model.predict(image)

if prediction[0][0] > 0.5:
    print("Prediction: Tumor Detected")
else:
    print("Prediction: No Tumor Detected")