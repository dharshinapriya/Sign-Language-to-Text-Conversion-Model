import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to SMN training data
DATA_DIR = 'D:\\yr\\Sign-Language-To-Text-Conversion-main\\dataSet\\trainingData'  # Change this to your actual data path

# Parameters
IMG_SIZE = 128

# Automatically determine the number of classes based on the subfolders in the dataset
NUM_CLASSES = len([folder for folder in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, folder))])

def load_data():
    data = []
    labels = []
    for label, folder in enumerate(sorted(os.listdir(DATA_DIR))):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    data.append(img)
                    labels.append(label)
    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(labels, NUM_CLASSES)
    return train_test_split(data, labels, test_size=0.2, random_state=42)

# Load dataset
X_train, X_test, y_train, y_test = load_data()

# Build model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),  # Explicit Input layer to avoid warning
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
if not os.path.exists("Models"):
    os.makedirs("Models")

model_json = model.to_json()
with open("Models/model-bw_smn.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("Models/model-bw_smn.weights.h5")  # Use correct extension

print("SMN model trained and saved successfully.")
