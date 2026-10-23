
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

tf.keras.utils.set_random_seed(42)

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

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train for 10 epochs
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

training_time = time.time() - start_time

# Evaluate on unseen test data
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\n--- MODEL 1 RESULTS ---")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# Save the trained model
model.save("model1_mlp.keras")

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model 1: Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model1_accuracy.png", dpi=150)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1: Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model1_loss.png", dpi=150)
plt.show()

# Compare performance at 5 and 10 epochs
print("\n--- EPOCH COMPARISON ---")
for epoch in [5, 10]:
    index = epoch - 1
    print(
        f"Epoch {epoch}: "
        f"Training accuracy = {history.history['accuracy'][index] * 100:.2f}%, "
        f"Validation accuracy = {history.history['val_accuracy'][index] * 100:.2f}%"
    )
