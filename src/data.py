from tensorflow.keras.datasets import fashion_mnist

def load_data(normalize=True):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    if normalize:
        x_train = x_train.astype("float32")/255.0
        x_test = x_test.astype("float32")/255.0
    
    class_names = [
        "T-shirt/top", "Trousers", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Anckle boot"
    ]
    return (x_train, y_train), (x_test, y_test), class_names