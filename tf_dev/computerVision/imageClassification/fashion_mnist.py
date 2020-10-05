"""
Developer: vkyprmr
Filename: fashion_mnist.py
Created on: 2020-10-5, Mo., 12:49:49
"""
"""
Modified by: vkyprmr
Last modified on: 2020-10-5, Mo., 16:53:43
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Loading the data
fm = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fm.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(x_train[5])
print(y_train[5])
print(x_train[5])
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()

# Normalizing the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model
model_name = 'fashion_mnist_simple'
layers = [
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
]
model = Sequential(layers=layers, name=model_name)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=5, verbose=1)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(test_loss, test_acc)

# Prediction
preds = model.predict(x_test)
pred_0 = preds[0]
print(f'Raw prediction: {pred_0},\nPredicted label: {np.argmax(pred_0)},\nActual label: {y_test[0]}')


# Plot predictions with actual images
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f'{class_names[predicted_label]} {100 * np.max(predictions_array):2.0f} {class_names[true_label]}',
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 3
num_cols = 2
num_images = num_rows * num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
idx = []
for i in range(num_images):
    r = np.random.randint(0,len(preds))
    idx.append(r)

for i in idx:
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, preds[i], y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, preds[i], y_test)

# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, preds[i], y_test, x_test)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, preds[i], y_test)

plt.tight_layout()
plt.show()
