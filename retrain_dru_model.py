# retrain_dru_model.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Dataset path and label setup
dataset_path = "dataSet/trainingData"
categories = ["D", "R", "U"]  # Label 0, 1, 2

data = []
labels = []

# 2. Load and preprocess images
for idx, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            image = image / 255.0  # normalize
            data.append(image)
            labels.append(idx)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# 3. Prepare dataset
data = np.array(data, dtype="float32").reshape(-1, 128, 128, 1)
labels = to_categorical(labels, num_classes=3)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 5. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')  # D, R, U
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test))

# 7. Save model
if not os.path.exists("Models"):
    os.makedirs("Models")

with open("Models/model-bw_dru.json", "w") as json_file:
    json_file.write(model.to_json())

model.save_weights("Models/model-bw_dru.weights.h5")

print(" DRU model retrained and saved to Models/model-bw_dru.*")

