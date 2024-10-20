document.addEventListener('DOMContentLoaded', () => {
    const animalRadios = document.querySelectorAll('input[name="animal"]');
    const animalImage = document.getElementById('animal-image');

    animalRadios.forEach(radio => {
        radio.addEventListener('change', fetchAnimalImage);
    });
});

function fetchAnimalImage() {
    const animal = document.querySelector('input[name="animal"]:checked').value;
    const animalImage = document.getElementById('animal-image');

    let imageUrl = `/images/${animal}.jpg`;

    animalImage.innerHTML = `<img src="${imageUrl}" alt="${animal}">`;
    
    const img = animalImage.querySelector('img');
    img.addEventListener('load', () => {
        console.log('Image loaded successfully');
    });
    img.addEventListener('error', () => {
        console.error('Error loading image');
        animalImage.innerHTML = '<p>Error loading image. Please try again.</p>';
    });
}

function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    const fileInfo = document.getElementById('file-info');

    if (!file) {
        fileInfo.innerHTML = '<p class="error">Please select a file.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            fileInfo.innerHTML = `<p class="error">${data.error}</p>`;
        } else {
            fileInfo.innerHTML = `
                <h3>File Analysis</h3>
                <p><strong>Name:</strong> ${data.name}</p>
                <p><strong>Size:</strong> ${formatFileSize(data.size)}</p>
                <p><strong>Type:</strong> ${data.type}</p>
                <p><strong>Potential Uses:</strong> ${suggestUses(data.type)}</p>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        fileInfo.innerHTML = '<p class="error">An error occurred while analyzing the file.</p>';
    });
}

function formatFileSize(sizeInBytes) {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = parseInt(sizeInBytes);
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function suggestUses(fileType) {
    const suggestions = {
        'application/pdf': 'Document sharing, e-books, printable materials',
        'image/jpeg': 'Web graphics, photo printing, social media sharing',
        'image/png': 'Web graphics, logos, transparent images',
        'video/mp4': 'Video streaming, social media content, presentations',
        'audio/mpeg': 'Music streaming, podcasts, audio books',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'Data analysis, financial reports, inventory management',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'Reports, essays, documentation',
        'application/zip': 'File compression, software distribution, backup archives'
    };

    return suggestions[fileType] || 'Various digital content purposes';
}
