
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

tf.keras.utils.set_random_seed(42)

# Load dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

def build_mlp(hidden_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28)))
    model.add(tf.keras.layers.Flatten())

    for _ in range(hidden_layers):
        model.add(
            tf.keras.layers.Dense(128, activation="relu")
        )

    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

results = []

for layers in [1, 2, 3]:
    print(f"\nTraining MLP with {layers} hidden layer(s)")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = build_mlp(layers)
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

    # Evaluate once on the held-out test set
    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results.append({
        "Hidden layers": layers,
        "Validation accuracy (epoch 5)": 
            history.history["val_accuracy"][4],
        "Validation accuracy (epoch 10)": 
            history.history["val_accuracy"][9],
        "Test accuracy": test_accuracy,
        "Test loss": test_loss,
        "Training time (seconds)": elapsed
    })

# Display and save results
df = pd.DataFrame(results)

print("\n--- HIDDEN LAYER EXPERIMENT RESULTS ---")
print(df.round(4).to_string(index=False))

df.to_csv("experiment_results.csv", index=False)

# Plot validation accuracy comparison
plt.figure(figsize=(8, 5))

for row in results:
    plt.plot(
        [5, 10],
        [
            row["Validation accuracy (epoch 5)"],
            row["Validation accuracy (epoch 10)"]
        ],
        marker="o",
        label=f'{row["Hidden layers"]} hidden layer(s)'
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Effect of Hidden Layers and Epochs")
plt.xticks([5, 10])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("experiment_comparison.png", dpi=150)
plt.show()
