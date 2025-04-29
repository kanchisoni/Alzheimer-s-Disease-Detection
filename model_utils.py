# model_utils.py
from keras.models import load_model
import joblib
import numpy as np
from keras.preprocessing import image

# load once at import
_feature_extractor = load_model('models/feature_extractor.h5')
_pca              = joblib.load('models/pca.pkl')
_svm              = joblib.load('models/svm_model.pkl')

def preprocess_image(img_path, target_size=(224,224)):
    """Load & normalize an image to numpy array."""
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_stage(img_path):
    # 1) extract CNN features
    feats = _feature_extractor.predict(preprocess_image(img_path))
    # 2) reduce dimensionality
    red  = _pca.transform(feats)
    # 3) classify with SVM
    cls  = _svm.predict(red)[0]
    # your mapping:
    labels = ['NonDemented','MildDemented','ModerateDemented','VeryMildDemented']
    return labels[int(cls)]
