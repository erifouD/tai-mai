document.addEventListener('DOMContentLoaded', () => {
    const btnRandom = document.getElementById('btn-random');
    const fileUpload = document.getElementById('file-upload');
    const imagePreview = document.getElementById('image-preview');
    const imagePlaceholder = document.getElementById('image-placeholder');
    const trueLabel = document.getElementById('true-label');
    const btnPredict = document.getElementById('btn-predict');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const errorMessage = document.getElementById('error-message');
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const probsContainer = document.getElementById('probabilities-bars');

    let currentFile = null;

    // Reset UI
    function resetUI() {
        resultContainer.style.display = 'none';
        errorMessage.style.display = 'none';
        trueLabel.textContent = '';
    }

    // Set image in UI
    function setImage(src) {
        imagePreview.src = src;
        imagePreview.style.display = 'block';
        imagePlaceholder.style.display = 'none';
        btnPredict.disabled = false;
        resetUI();
    }

    // Convert base64 to File object
    function dataURLtoFile(dataurl, filename) {
        const arr = dataurl.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
    }

    // Fetch random image
    btnRandom.addEventListener('click', async () => {
        try {
            btnRandom.disabled = true;
            btnRandom.textContent = 'Loading...';
            
            const response = await fetch('/get_random_image');
            const data = await response.json();
            
            setImage(data.image);
            currentFile = dataURLtoFile(data.image, `test_img_${data.index}.png`);
            trueLabel.textContent = `True Label (from dataset): ${data.label}`;
            
        } catch (error) {
            console.error('Error fetching random image:', error);
            showError('Failed to fetch random image. Is the server running?');
        } finally {
            btnRandom.disabled = false;
            btnRandom.textContent = 'Get Random Test Image';
        }
    });

    // Handle file upload
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            currentFile = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = (e) => {
                setImage(e.target.result);
                trueLabel.textContent = `Uploaded File: ${currentFile.name}`;
            };
            
            reader.readAsDataURL(currentFile);
        }
    });

    // Handle prediction
    btnPredict.addEventListener('click', async () => {
        if (!currentFile) return;

        try {
            // UI state
            btnPredict.disabled = true;
            loading.style.display = 'block';
            resultContainer.style.display = 'none';
            errorMessage.style.display = 'none';

            // Prepare form data
            const formData = new FormData();
            formData.append('file', currentFile);

            // Send to server
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Show success
                predClass.textContent = data.class;
                predConf.textContent = data.confidence;
                
                // Render probability bars
                probsContainer.innerHTML = '';
                
                // Sort probabilities descending
                const sortedProbs = Object.entries(data.probabilities)
                    .sort((a, b) => b[1] - a[1]);

                sortedProbs.forEach(([className, prob]) => {
                    const percent = (prob * 100).toFixed(1);
                    const isTop = className === data.class;
                    
                    const row = document.createElement('div');
                    row.className = 'prob-row';
                    row.innerHTML = `
                        <div class="prob-label" style="font-weight: ${isTop ? '700' : '400'}">${className}</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${percent}%; background-color: ${isTop ? 'var(--primary-color)' : 'var(--text-muted)'}"></div>
                        </div>
                        <div class="prob-percent">${percent}%</div>
                    `;
                    probsContainer.appendChild(row);
                });

                // Small delay to allow CSS transitions to trigger
                setTimeout(() => {
                    resultContainer.style.display = 'block';
                }, 10);
                
            } else {
                showError(data.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Prediction error:', error);
            showError('Failed to connect to the server for prediction.');
        } finally {
            btnPredict.disabled = false;
            loading.style.display = 'none';
        }
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
    }
});
