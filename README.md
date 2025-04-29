 Alzheimer's Stage Classification Web App
This web application uses deep learning and machine learning models to classify the stage of Alzheimer's Disease from an uploaded MRI brain scan image.

📌 Project Summary
The project integrates a CNN-based feature extractor, PCA for dimensionality reduction, and an SVM classifier to identify one of four Alzheimer's stages:

NonDemented

VeryMildDemented

MildDemented

ModerateDemented

🚀 Features
Upload MRI scan images (JPEG, PNG).

Predict Alzheimer's stage using pre-trained models.

Display prediction confidence.

Real-time web interface using Flask.

🧠 Tech Stack
Python

Flask

Keras (TensorFlow backend)

scikit-learn

Joblib

HTML/CSS (Jinja2 templates)

Bootstrap (optional for frontend design)

📁 Project Structure
csharp
Copy
Edit
project/
├── app.py                   # Main Flask application
├── models/
│   ├── feature_extractor.h5 # CNN feature extractor
│   ├── pca.pkl              # Trained PCA model
│   └── svm_model.pkl        # Trained SVM model
├── static/
│   └── uploads/             # Stores uploaded images
├── templates/
│   └── index.html           # Upload form UI
├── README.md
└── requirements.txt         # Python dependencies
⚙️ How It Works
Image Upload: User uploads an MRI scan.

Preprocessing: The image is resized and normalized to 224x224.

Feature Extraction: CNN extracts image features.

PCA: Reduces features to smaller vectors.

SVM Classifier: Classifies based on reduced features.

Result: Returns stage and confidence level.

🛠️ Setup Instructions
Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/alzheimers-stage-classifier.git
cd alzheimers-stage-classifier
Step 2: Create Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Step 4: Add Trained Models
Place the following files in the models/ folder:

feature_extractor.h5

pca.pkl

svm_model.pkl

Step 5: Run the App
bash
Copy
Edit
python app.py
Then visit http://127.0.0.1:5000 in your browser.

🖼️ Sample Output
yaml
Copy
Edit
Predicted Stage: VeryMildDemented
Confidence: 94.3%
🔍 Notes
Maximum file size: 16 MB

Supported formats: PNG, JPG, JPEG

🚧 Future Improvements
Grad-CAM visualization support

Batch image upload support

Cloud deployment (Render, AWS, etc.)

Medical disclaimer and data privacy info
