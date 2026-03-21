document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const scanBtn = document.getElementById('scan-btn');
    const preview = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');

    let currentFile = null;

    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            currentFile = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                placeholder.style.display = 'none';
                scanBtn.disabled = false;
                results.style.display = 'none';
            };
            reader.readAsDataURL(currentFile);
        }
    });

    scanBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        scanBtn.disabled = true;
        loading.style.display = 'block';
        results.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const resp = await fetch('/predict', { method: 'POST', body: formData });
            const data = await resp.json();

            if (resp.ok) {
                // Анимация задержки для "тяжелой" обработки
                setTimeout(() => {
                    loading.style.display = 'none';
                    
                    document.getElementById('type-result').innerText = data.type.class;
                    document.getElementById('type-conf').innerText = data.type.confidence;
                    document.getElementById('type-bar').style.width = data.type.confidence;

                    document.getElementById('model-result').innerText = data.model.class;
                    document.getElementById('model-conf').innerText = data.model.confidence;
                    document.getElementById('model-bar').style.width = data.model.confidence;

                    results.style.display = 'block';
                    scanBtn.disabled = false;
                }, 800);
            } else {
                alert("ОШИБКА СИСТЕМЫ: " + (data.error || "Сбой классификации"));
                loading.style.display = 'none';
                scanBtn.disabled = false;
            }
        } catch (err) {
            alert("КРИТИЧЕСКАЯ ОШИБКА СВЯЗИ С СЕРВЕРОМ");
            loading.style.display = 'none';
            scanBtn.disabled = false;
        }
    });
});
