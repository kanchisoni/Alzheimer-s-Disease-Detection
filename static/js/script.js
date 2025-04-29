const imageUpload = document.getElementById('imageUpload');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const previewImage = document.getElementById('previewImage');
const stageText = document.getElementById('stage');
const confidenceText = document.getElementById('confidence');
const descriptionText = document.getElementById('description');

imageUpload.addEventListener('change', function() {
    if (imageUpload.files.length > 0) {
        analyzeBtn.disabled = false;
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(imageUpload.files[0]);
    } else {
        analyzeBtn.disabled = true;
    }
});

analyzeBtn.addEventListener('click', function() {
    if (imageUpload.files.length === 0) {
        alert('Please upload an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', imageUpload.files[0]);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        stageText.textContent = data.stage;
        confidenceText.textContent = data.confidence;
        descriptionText.textContent = data.description;
        previewImage.src = data.image_url;
        resultSection.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing the image.');
    });
});
