import streamit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("Fashion MNIST Classifier Project")

model = tf.keras.models.load_model("artifacts/model")
labels = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Anckle boot"]

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convery("L")
    img = ImageOps.fit(img, (28,28))
    arr = np.array(img).astype("flopat32")/255.0
    arr = np.expand_dims(arr, axis=(0,-1))

    preds = model.predict(arr)[0]
    st.write("Prediction:", labels[np.argmax(preds)])
    st.bar_chart(preds)
