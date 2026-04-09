# Import Library yang digunakan
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras import layers, models
from keras.datasets import mnist

# Load dataset and split data train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Read data
plt.imshow(x_train[0], cmap='gray')

# Reshipe and Normalize data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train/255
x_test = x_test/255

# Check Class Data
set(y_train.tolist())

# Create CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
]
)

# Summary Data
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training Model
hasil = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2)

# Evaluate using Matplotlib
y1 = hasil.history['accuracy']
y2 = hasil.history['val_accuracy']
plt.plot(range(len(y1)), y1, 'g',
         range(len(y2)), y2, 'b')

plt.show()

# Test Accuracy model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy = ", test_accuracy)

# Upload test image
img = cv2.imread('angka enam.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')   

img.shape

img = cv2.resize(img, (28, 28))
plt.imshow(img, cmap='gray')

img = img / 255
img = img.reshape((1, 28, 28, 1))
img.shape

pred = model.predict(img)
pred = np.argmax(pred)
print('Prediksi Gambar, Angka = ', pred)

# Import additional required libraries
import pickle
import os
from PIL import Image
import time

# ... [Previous code remains the same until after model evaluation] ...

# Save the trained model and training history
def save_training_results(model, history, accuracy):
    # Create directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save the model
    model.save('mnist_cnn.h5')
    
    # Save training history and accuracy
    training_data = {
        'history': history.history,
        'test_accuracy': accuracy
    }
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(training_data, f)
        
    print("Model and training results saved successfully.")

# Call the save function
save_training_results(model, hasil, test_accuracy)