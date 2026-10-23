
import tensorflow as tf
import pandas as pd
import numpy as np

# Load the test dataset
(_, _), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

X_test = X_test.astype("float32") / 255.0

models = [
    ("Basic MLP", "model1_mlp.keras", False),
    ("Deep MLP", "model2_deep.keras", False),
    ("CNN", "model3_cnn.keras", True),
]

results = []

for name, path, is_cnn in models:
    model = tf.keras.models.load_model(path)

    X = X_test[..., None] if is_cnn else X_test

    loss, accuracy = model.evaluate(X, y_test, verbose=0)

    results.append({
        "Model": name,
        "Test accuracy (%)": round(accuracy * 100, 2),
        "Test loss": round(loss, 4)
    })

df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(df.to_string(index=False))

df.to_csv("final_model_comparison.csv", index=False)

best = df.loc[df["Test accuracy (%)"].idxmax()]
print(
    f"\nBest model by test accuracy: {best['Model']} "
    f"({best['Test accuracy (%)']:.2f}%)"
)
