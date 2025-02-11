from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io
import logging
import threading
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create required directories if they don't exist
if os.environ.get('RENDER') or os.environ.get('RAILWAY_ENVIRONMENT'):
    # Use /tmp for cloud deployments
    UPLOAD_FOLDER = '/tmp/uploads'
    PROCESSED_FOLDER = '/tmp/processed'
else:
    # Local development paths
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload and processed directories exist at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Add error handling for file operations
@app.errorhandler(Exception)
def handle_error(error):
    app.logger.error(f"An error occurred: {str(error)}")
    return jsonify({"error": "An error occurred while processing the image. Please try again."}), 500

# Load OpenCV's pre-trained classifiers
try:
    cascade_path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))

    if face_cascade.empty():
        app.logger.error("Error: Could not load face cascade classifier")
    if eye_cascade.empty():
        app.logger.error("Error: Could not load eye cascade classifier")
except Exception as e:
    app.logger.error(f"Error loading classifiers: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'}

def apply_ai_safe_filter(image, parameters=None):
    """Apply AI-safe filter to the image"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Basic image processing without face detection
        # Convert to HSV for better color manipulation
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
            
        # Apply basic noise and pattern
        h, w = img_array.shape[:2]
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        result = cv2.add(img_array, noise)
        
        # Add a subtle pattern
        pattern = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                pattern[i:i+4, j:j+4] = 255
        
        pattern = cv2.merge([pattern, pattern, pattern])
        result = cv2.addWeighted(result, 0.9, pattern, 0.1, 0)
        
        # Convert back to PIL Image
        return Image.fromarray(result)
        
    except Exception as e:
        app.logger.error(f"Error in apply_ai_safe_filter: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Create a secure filename and ensure directories exist
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        
        # Save the uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process the image
        try:
            with Image.open(upload_path) as img:
                processed_img = apply_ai_safe_filter(img)
                processed_filename = f"processed_{filename}"
                processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                processed_img.save(processed_path)
                
                return jsonify({
                    'filename': processed_filename,
                    'message': 'Image processed successfully'
                })
                
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Error processing image. Please try again.'}), 500
            
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Error uploading file'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(PROCESSED_FOLDER, filename),
            as_attachment=True
        )
    except Exception as e:
        app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Error downloading file'}), 500

def cleanup_old_files():
    """Remove files older than 1 hour from uploads and processed folders"""
    while True:
        try:
            current_time = datetime.now()
            for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if current_time - file_time > timedelta(hours=1):
                            try:
                                os.remove(filepath)
                                logger.info(f"Removed old file: {filepath}")
                            except Exception as e:
                                logger.error(f"Error removing file {filepath}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
        time.sleep(3600)  # Run every hour

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)
