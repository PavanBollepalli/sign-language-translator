import cv2
import numpy as np
import os
import json
from model import ASLModel

class ASLInference:
    def __init__(self, model_path, class_map_path=None, img_size=(64, 64)):
        self.model = ASLModel()
        self.model.load(model_path)
        self.img_size = img_size
        
        # Load class mapping if provided
        if class_map_path and os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                self.classes = json.load(f)
        else:
            # Default ASL alphabet
            self.classes = {i: chr(65 + i) for i in range(26)}  # A-Z
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize
        resized = cv2.resize(image, self.img_size)
        
        # Apply histogram equalization to improve contrast
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Normalize
        normalized = enhanced_img / 255.0
        
        return normalized
    
    def predict(self, image):
        """Predict the sign in the image"""
        processed_img = self.preprocess_image(image)
        prediction = self.model.predict(processed_img)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        
        # Get the predicted letter
        letter = self.classes.get(class_idx, "Unknown")
        
        return {
            "letter": letter,
            "confidence": float(confidence),
            "class_idx": int(class_idx)
        }
    
    def predict_from_file(self, image_path):
        """Predict sign from an image file"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.predict(img)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on ASL images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image file to predict")
    parser.add_argument("--class_map", type=str, default=None,
                        help="Path to class mapping JSON file")
    
    args = parser.parse_args()
    
    # Run inference
    asl_inference = ASLInference(args.model_path, args.class_map)
    result = asl_inference.predict_from_file(args.image_path)
    
    print(f"Predicted letter: {result['letter']}")
    print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
