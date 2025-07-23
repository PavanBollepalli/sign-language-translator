import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class ASLDataLoader:
    def __init__(self, data_dir, img_size=(64, 64), test_size=0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_size = test_size
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        self.classes = self._get_classes()
        if not self.classes:
            raise ValueError(f"No class directories found in {data_dir}. Expected subdirectories for each class.")
            
        self.num_classes = len(self.classes)
        
    def _get_classes(self):
        classes = sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])
        print(f"Found {len(classes)} class directories: {classes}")
        return classes
    
    def load_data(self):
        images = []
        labels = []
        
        print(f"Loading images from {len(self.classes)} classes...")
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"Warning: No images found in class directory {class_dir}")
                continue
                
            print(f"Loading {len(image_files)} images from class '{class_name}'")
            loaded_count = 0
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image file {img_path}")
                        continue
                        
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(idx)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
            
            print(f"Successfully loaded {loaded_count} images from class '{class_name}'")
        
        if not images:
            raise ValueError("No images were loaded from the dataset. Please check the dataset directory structure and image formats.")
            
        print(f"Total images loaded: {len(images)}")
        print(f"Total labels: {len(labels)}")
        
        X = np.array(images)
        y = np.array(labels, dtype=np.int32)  # Explicitly set dtype to int32
        
        # Convert labels to one-hot encoding
        y_onehot = np.zeros((y.size, self.num_classes))
        y_onehot[np.arange(y.size), y] = 1
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=self.test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test, self.classes
