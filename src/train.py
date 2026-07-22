from preprocess import load_data
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
print("Loading training data...")
X_train, y_train = load_data("dataset/train", limit=100)
print("Training Images:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Loading testing data...")
X_test, y_test = load_data("dataset/test", limit=50)
print("Testing Images:", X_test.shape)
print("Testing Labels:", y_test.shape)
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting Training...\n")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=16
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

model.save("models/brain_tumor_model.keras")

print("\nModel Saved!")