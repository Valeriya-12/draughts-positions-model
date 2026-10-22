"""Train a MobileNetV2 model to classify draughts-board cells."""
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from pathlib import Path

DATASET_DIR = Path("dataset")
EXPORT_DIR = Path("export_model")
KERAS_MODEL_PATH = "checkers_mobilenetv2.h5"

IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 20

def train_and_export():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder {DATASET_DIR} was not found. Create the dataset before training."
        )

    print("--- 1. Loading and splitting the dataset ---")

    # 1. Load the training data (80%).
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    # 2. Load the validation data (20%).
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected classes: {num_classes} -> {class_names}")

    # Optimize data-loading performance.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n--- 2. Building the model architecture ---")

    # Augmentation layers (small random shifts, rotations, and contrast changes).
    data_augmentation = models.Sequential([
        layers.RandomRotation(0.05),      # Rotate by up to approximately 18 degrees.
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1)
    ])

    # Base MobileNetV2 model without its top classification layer.
    base_model = MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze the convolutional layers.

    # Assemble the final network.
    inputs = layers.Input(shape=(64, 64, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)  # Normalize values from [0, 255] to [0, 1].
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    print("\n--- 3. Training the model ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save the model in the standard Keras format (.h5).
    model.save(KERAS_MODEL_PATH)

    print(f"\nKeras model saved successfully to: {KERAS_MODEL_PATH}")