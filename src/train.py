from .data import load_data
from .model import build_model

(x_train, y_train), (x_test, y_test), class_names = load_data()
model = build_model()

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=128
)

model.save("artifacts/model")
