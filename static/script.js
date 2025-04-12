document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const processingContainer = document.getElementById('processingContainer');
    const spinner = processingContainer.querySelector('.spinner');
    const processingText = processingContainer.querySelector('.processing-text');
    const resultContainer = document.getElementById('resultContainer');
    const predictionResult = document.getElementById('predictionResult');
    const fileInfo = document.getElementById('fileInfo');

    processingContainer.style.display = 'none';
    resultContainer.style.display = 'none';

    browseBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadArea.addEventListener('click', function(e) {
        if (!e.target.closest('#browseBtn')) {
            fileInput.click();
        }
    });

    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file.');
            return;
        }

        fileInfo.textContent = `Selected file: ${file.name} (${formatFileSize(file.size)})`;

        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        
        reader.onloadend = function() {
            resultContainer.style.display = 'none';
        
            processingContainer.style.display = 'block';
            spinner.style.display = 'block';
            processingText.textContent = 'Processing your image...';
        
            uploadAndPredict(file);
        };
        
        reader.readAsDataURL(file);
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                predictionResult.textContent = data.caption;
                resultContainer.style.display = 'block';
            } else {
                predictionResult.textContent = `Error: ${data.error}`;
                resultContainer.style.display = 'block';
            }
        } catch (error) {
            predictionResult.textContent = `Error: ${error.message}`;
            resultContainer.style.display = 'block';
        } finally {
            spinner.style.display = 'none';
            processingText.textContent = 'Processing complete!';
            
            setTimeout(() => {
                processingContainer.style.display = 'none';
                processingText.textContent = 'Processing your image...';
            }, 3000);
        }
    }
});