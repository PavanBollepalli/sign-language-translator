import cv2
import numpy as np
import argparse
from inference import ASLInference

def main(args):
    # Initialize the ASL inference model
    asl_inference = ASLInference(args.model_path, args.class_map)
    
    # Start webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set up region of interest (ROI)
    roi_size = 224
    roi_x = 100
    roi_y = 100
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size), 
                     (0, 255, 0), 2)
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Make prediction
        if roi.size != 0:  # Check if ROI is not empty
            result = asl_inference.predict(roi)
            
            # Display prediction
            text = f"{result['letter']} ({result['confidence']:.2f})"
            cv2.putText(frame, text, (roi_x, roi_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        # Display the resulting frame
        cv2.imshow('ASL Sign Detection', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run webcam demo for ASL detection")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--class_map", type=str, default=None,
                        help="Path to class mapping JSON file")
    
    args = parser.parse_args()
    main(args)
