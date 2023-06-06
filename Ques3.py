import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist

# Set random seeds for reproducibility
tf.random.set_seed(42)

# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
test_images = test_images[..., tf.newaxis].astype("float32") / 255.0

# Create the CNN model
model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=2)

# Evaluate the model
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)

print(f"Validation accuracy: {accuracy*100:.2f}%")
