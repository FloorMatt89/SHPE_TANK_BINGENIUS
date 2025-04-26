import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
DATASET_DIR = "archive/images/images"

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

image_paths = []
labels = []
class_indices = {}

for idx, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_dir):
        class_indices[class_name] = idx
        for root, _, files in os.walk(class_dir):  # 🔁 walk through subfolders
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, fname)
                    image_paths.append(full_path)
                    labels.append(idx)

num_classes = len(class_indices)
classes = sorted(class_indices.keys())
print(f" Found {len(image_paths)} images across {len(class_indices)} classes")


train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, stratify=labels, test_size=0.3, random_state=42)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, stratify=temp_labels, test_size=1/3, random_state=42)


def preprocess_input_resnet(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img



def process_path(path, label, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    if augment:
        img = data_augmentation(img)

    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, tf.one_hot(label, depth=num_classes)

def build_dataset(paths, labels, augment=False):
    def map_fn(path, label):
        return process_path(path, label, augment)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(train_paths, train_labels, augment=True)
val_ds = build_dataset(val_paths, val_labels, augment=False)
test_ds = build_dataset(test_paths, test_labels, augment=False)


inputs = Input(shape=IMG_SIZE + (3,))
base_model = ResNet50(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False

x = base_model(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
outputs = Dense(units=num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


model.summary()
model.fit(train_ds, epochs=15, validation_data=val_ds)
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n Test Accuracy: {test_acc:.4f}")

model.save("waste_classifier.h5")
