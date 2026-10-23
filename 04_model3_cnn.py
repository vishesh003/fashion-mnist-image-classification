
import tensorflow as tf
import matplotlib.pyplot as plt
import time

tf.keras.utils.set_random_seed(42)

# Load and normalize Fashion-MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# CNN expects images in (height, width, channels) format
X_train = X_train[..., None]
X_test = X_test[..., None]

# Model 3: Convolutional Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, kernel_size=3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(
        64, kernel_size=3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

start = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

elapsed = time.time() - start

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\n--- MODEL 3 RESULTS ---")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Training time: {elapsed:.2f} seconds")

model.save("model3_cnn.keras")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model 3: CNN Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model3_accuracy.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 3: CNN Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model3_loss.png", dpi=150)
plt.show()
