import os
import warnings
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras import models
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore")

# Set dataset path manually
image_dir = Path("data")  # Make sure the dataset is in this folder
if not image_dir.exists():
    raise FileNotFoundError("Dataset not found. Please download and extract it to the project folder.")

# Prepare dataset
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg'))
labels = [x.parent.name for x in filepaths]

image_df = pd.DataFrame({
    "Filepath": [str(x) for x in filepaths],
    "Label": labels
})

# Show sample images
categories = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for col, category in enumerate(categories):
    samples = image_df[image_df['Label'] == category].sample(4)
    for row in range(4):
        img = plt.imread(samples.iloc[row]['Filepath'])
        axes[row, col].imshow(img)
        axes[row, col].set_title(category)
        axes[row, col].axis('off')
plt.tight_layout()
plt.show()

# Prepare data directories
for split in ['train', 'validation']:
    for category in categories:
        os.makedirs(f"data/{split}/{category}", exist_ok=True)

train_data, val_data = train_test_split(image_df, test_size=0.2, stratify=image_df['Label'], random_state=42)

def move_images(df, split):
    for _, row in df.iterrows():
        src_path = Path(row['Filepath'])
        dest_path = Path(f"data/{split}/{row['Label']}/{src_path.name}")
        
        # Check if the destination file exists, and only copy if it's different
        if not dest_path.exists():
            shutil.copy(src_path, dest_path)
        else:
            print(f"File already exists: {dest_path.name}, skipping copy.")

move_images(train_data, 'train')
move_images(val_data, 'validation')

print("Data split and organized.")

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_images = train_datagen.flow_from_directory('data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_images = train_datagen.flow_from_directory('data/validation', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Model definition
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
feature_extractor = models.Sequential([base_model, GlobalAveragePooling2D()])

# Feature extraction
def extract_features(generator, model):
    features, labels = [], []
    for img_batch, label_batch in generator:
        feat = model.predict(img_batch)
        features.extend(feat)
        labels.extend(np.argmax(label_batch, axis=1))
        if len(features) >= generator.n:
            break
    return np.array(features), np.array(labels)

X_train, y_train = extract_features(train_images, feature_extractor)
X_val, y_val = extract_features(validation_images, feature_extractor)

# Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# PCA
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_resampled)
X_val_pca = pca.transform(X_val)

# SVM training
svm_model_pca = SVC(kernel='rbf', C=1.0, class_weight='balanced')
svm_model_pca.fit(X_train_pca, y_train_resampled)

# Save models
os.makedirs("models", exist_ok=True)
feature_extractor.save("models/feature_extractor.h5")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(svm_model_pca, "models/svm_model.pkl")
print("Models saved.")

# Prediction
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_alzheimers_stage(img_path):
    try:
        loaded_model = models.load_model("models/feature_extractor.h5")
        pca = joblib.load("models/pca.pkl")
        svm_model = joblib.load("models/svm_model.pkl")

        img_array = preprocess_image(img_path)
        features = loaded_model.predict(img_array)
        reduced = pca.transform(features)
        prediction = svm_model.predict(reduced)

        categories = ['NonDemented', 'MildDemented', 'ModerateDemented', 'VeryMildDemented']
        print("Prediction:", categories[prediction[0]])
    except Exception as e:
        print("Error during prediction:", e)

# Use your own image file path here
img_path = "test_image.jpg"  # Replace with actual image path
predict_alzheimers_stage(img_path)
