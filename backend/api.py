import os
import base64
import io
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import ASLInference

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize ASL inference model
MODEL_PATH = os.environ.get('ASL_MODEL_PATH', 'models/asl_model.h5')
CLASS_MAP_PATH = os.environ.get('ASL_CLASS_MAP_PATH', 'models/class_map.json')

# Initialize model only if it exists
inference = None
if os.path.exists(MODEL_PATH):
    inference = ASLInference(MODEL_PATH, CLASS_MAP_PATH)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": inference is not None})

@app.route('/api/detect', methods=['POST'])
def detect_sign():
    if inference is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Decode base64 image
        image_data = request.json['image']
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        img_bytes = base64.b64decode(image_data)
        img_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Apply additional preprocessing
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply slight blur to reduce noise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Run prediction
        result = inference.predict(img)
        
        # Apply additional filtering
        if result["confidence"] < 0.3:  # Very low confidence
            result["letter"] = "Unknown"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    if 'text' not in request.json:
        return jsonify({"error": "No text provided"}), 400
    
    text = request.json['text']
    # This is a placeholder for actual translation logic
    # In a real application, you would convert the text to ASL signs
    result = {
        "text": text,
        "signs": [char.upper() for char in text if char.isalpha()]
    }
    
    return jsonify(result)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Start ASL detection API server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind the server to")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server to")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to the trained model file")
    parser.add_argument("--class_map", type=str, default=CLASS_MAP_PATH,
                        help="Path to class mapping JSON file")
    
    args = parser.parse_args()
    
    # Update model path if provided
    if args.model_path != MODEL_PATH:
        MODEL_PATH = args.model_path
        if os.path.exists(MODEL_PATH):
            inference = ASLInference(MODEL_PATH, args.class_map)
    
    app.run(host=args.host, port=args.port, debug=True)
