import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
IMG_SIZE = 128
model = load_model("models/brain_tumor_model.keras")
st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader(
    "Upload an MRI Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded MRI")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        st.success("NO Tumor Detected")
    else:
        st.success("Tumor Detected")