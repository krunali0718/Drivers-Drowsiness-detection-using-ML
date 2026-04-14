import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("Loading Deep Learning Framework (TensorFlow)...")

# 1. Setup Local Dataset Path
DATASET_PATH = r"d:\SECOND YEAR\SEM2\AD\driver_drowsiness_project\dataset"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

print(f"Reading pure images directly from: {DATASET_PATH}")
print("This breaks the accuracy limit by using Deep Convolutional Neural Networks (CNNs) with Balanced Data and Augmentation.")

# 2. Count Images to Automatically Balance Classes
drowsy_dir = os.path.join(DATASET_PATH, 'Drowsy')
awake_dir = os.path.join(DATASET_PATH, 'Non Drowsy')

# Count how many files are in each directory mapping
drowsy_count = len(os.listdir(drowsy_dir)) if os.path.exists(drowsy_dir) else 0
awake_count = len(os.listdir(awake_dir)) if os.path.exists(awake_dir) else 0

total_images = drowsy_count + awake_count
print(f"\n[Dataset Status] Drowsy Images: {drowsy_count} | Awake (Non Drowsy) Images: {awake_count}")

if total_images == 0:
    raise FileNotFoundError(f"Could not find images in {DATASET_PATH}. Please check folder structure.")

# Calculate exact mathematical Class Weights (Drowsy = Awake Balance)
# We assume 'Drowsy' corresponds to Class 0 and 'Non Drowsy' to Class 1, or vice versa.
# Keras image_dataset_from_directory sorts classes alphabetically!
# Alphabetical sort: 'Drowsy' -> 0, 'Non Drowsy' -> 1
weight_for_0 = (1 / drowsy_count) * (total_images / 2.0)
weight_for_1 = (1 / awake_count) * (total_images / 2.0)

class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"[Balance Configuration] Calculated Class Weights: Class 0 (Drowsy) = {weight_for_0:.2f}, Class 1 (Awake) = {weight_for_1:.2f}\n")

# 3. Load Image Dataset Automatically from Folders
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Detected Classes from folders: {class_names}")

# Configure dataset for performance (Prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# 4. Build a High-Accuracy CNN Pipeline with Built-in Data Augmentation
print("\nBuilding Convolutional Neural Network (CNN)...")

# Define Data Augmentation specifically inside the Model Graph
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),            # Flip left-right
  layers.RandomRotation(0.1),                  # Rotate randomly ±10%
  layers.RandomBrightness(factor=0.1)         # Randomly adjust brightness ±10%
], name="data_augmentation")

model = models.Sequential([
    # Inputs and Augmentation
    layers.InputLayer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    data_augmentation,
    
    # Rescaling Pixel Values (0-255 -> 0-1)
    layers.Rescaling(1./255),
    
    # Convolutional Block 1
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Convolutional Block 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Classification Head
    layers.Dropout(0.4), # Prevent Overfitting
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names))
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model (Deep Training)
print("\nSTARTING HEAVY DEEP LEARNING TRAINING...")
print("This may take 1+ hour depending on hardware. Sit back and relax.")

# Run for 20 Epochs for max accuracy
epochs = 20
SAVE_NAME = "cnn_model_augmented.keras"

# Create a callback to save the model after every successful epoch (Protects against crashes!)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=SAVE_NAME,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight=class_weights,  # Applied explicit balance here!
    callbacks=[checkpoint_callback]
)

# 6. Save the Final Model (Native Keras format to prevent serialization crash)
model.save(SAVE_NAME)

print(f"\nSUCCESS! High-Accuracy Deep Learning CNN has been saved as '{SAVE_NAME}'.")
print("Your Drowsiness Detection System is now perfectly balanced and trained via Computer Vision!")
