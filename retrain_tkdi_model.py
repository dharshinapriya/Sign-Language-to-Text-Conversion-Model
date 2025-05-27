# retrain_tkdi_model.py

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Define dataset path and categories
dataset_path = "dataSet/trainingData"
categories = ["T", "K", "D", "I"]

data = []
labels = []

# 2. Load and preprocess images
for idx, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            data.append(img)
            labels.append(idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# 3. Prepare data
data = np.array(data, dtype="float32").reshape(-1, 128, 128, 1)
labels = to_categorical(labels, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 4. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')  # 4 output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train model
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test))

# 6. Save model
if not os.path.exists("Models"):
    os.makedirs("Models")

with open("Models/model-bw_tkdi.json", "w") as json_file:
    json_file.write(model.to_json())

model.save_weights("Models/model-bw_tkdi.weights.h5")

print("✅ TKDI model retrained and saved to Models/model-bw_tkdi.*")