// JavaScript for AI Skin Disease Detection App

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadRadio = document.getElementById('upload');
    const cameraRadio = document.getElementById('camera');
    const uploadSection = document.getElementById('uploadSection');
    const cameraSection = document.getElementById('cameraSection');
    const imageUpload = document.getElementById('imageUpload');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const predictionResult = document.getElementById('predictionResult');
    const treatmentPlan = document.getElementById('treatmentPlan');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatContainer = document.getElementById('chatContainer');

    let currentImage = null;
    let stream = null;

    // Toggle input methods
    uploadRadio.addEventListener('change', function() {
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
        stopCamera();
        currentImage = null;
        analyzeBtn.disabled = true;
    });

    cameraRadio.addEventListener('change', function() {
        uploadSection.style.display = 'none';
        cameraSection.style.display = 'block';
        startCamera();
    });

    // File upload handling
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImage = e.target.result;
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    // Camera functions
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
        } catch (err) {
            alert('Không thể truy cập camera: ' + err.message);
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    // Capture image from camera
    captureBtn.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        currentImage = canvas.toDataURL('image/jpeg');
        analyzeBtn.disabled = false;
        stopCamera();
    });

    // Analyze image
    analyzeBtn.addEventListener('click', async function() {
        if (!currentImage) return;

        loading.style.display = 'block';
        result.style.display = 'none';
        analyzeBtn.disabled = true;

        try {
            // Convert base64 to blob
            const response = await fetch(currentImage);
            const blob = await response.blob();

            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            // Send to server
            const predictResponse = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await predictResponse.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Display results
            displayResults(data);
            result.style.display = 'block';

        } catch (error) {
            alert('Lỗi phân tích: ' + error.message);
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Display image
        const imgHtml = `<img src="data:image/jpeg;base64,${data.image}" class="img-fluid rounded mb-3" style="max-height: 200px;">`;

        // Display prediction
        const predictionHtml = `
            <div class="prediction-result">
                <h5><i class="fas fa-stethoscope"></i> Kết quả: ${data.prediction}</h5>
                <div class="confidence-bar" style="width: ${data.confidence * 100}%"></div>
                <p><strong>Độ tin cậy: ${(data.confidence * 100).toFixed(2)}%</strong></p>
            </div>
        `;

        // Display probabilities
        let probHtml = '<div class="probabilities"><h6>Xác suất cho từng loại:</h6>';
        for (const [className, prob] of Object.entries(data.probabilities)) {
            probHtml += `<div class="probability-item"><span>${className}</span><span>${(prob * 100).toFixed(2)}%</span></div>`;
        }
        probHtml += '</div>';

        predictionResult.innerHTML = imgHtml + predictionHtml + probHtml;
        treatmentPlan.innerHTML = data.treatment_plan;
    }

    // Chat functionality
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage('user', message);
        chatInput.value = '';

        // Send to server
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            if (data.error) {
                addMessage('assistant', 'Lỗi: ' + data.error);
            } else {
                addMessage('assistant', data.response);
                if (data.citations) {
                    addMessage('assistant', data.citations);
                }
            }
        } catch (error) {
            addMessage('assistant', 'Lỗi kết nối: ' + error.message);
        }
    }

    function addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.innerHTML = `<strong>${type === 'user' ? 'Bạn:' : 'Trợ lý:'}</strong> ${content}`;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Initialize
    uploadRadio.checked = true;
    uploadSection.style.display = 'block';
    cameraSection.style.display = 'none';
});