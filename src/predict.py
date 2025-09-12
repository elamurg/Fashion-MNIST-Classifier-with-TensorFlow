import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(path):
    img = Image.open(path).convert("L").resize((28,28))
    arr = np.array(img).astype("float32")/255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

model = tf.keras.models.load_model("artifacts/model")
x = preprocess_image("sample.png")

preds = model.predict(x)
print("Predicted class:", preds.argmax())

