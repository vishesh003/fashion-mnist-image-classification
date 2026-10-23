
# Load and normalize Fashion-MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Build Model 1: Basic Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
