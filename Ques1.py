import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# CNN Model 1
model1 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train Model 1
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# CNN Model 2
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train Model 2
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# CNN Model 3
model3 = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train Model 3
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Comparison table
models = ['CNN Model 1', 'CNN Model 2', 'CNN Model 3']
train_acc = [model1.history.history['accuracy'][-1],
             model2.history.history['accuracy'][-1],
             model3.history.history['accuracy'][-1]]
test_acc = [model1.history.history['val_accuracy'][-1],
            model2.history.history['val_accuracy'][-1],
            model3.history.history['val_accuracy'][-1]]

comparison_table = pd.DataFrame({'Model': models, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc})
print(comparison_table)
