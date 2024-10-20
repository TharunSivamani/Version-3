from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer to the beginning
        file_type = file.content_type
        
        return jsonify({
            'name': filename,
            'size': f"{file_size} bytes",
            'type': file_type
        })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)
