"""
Script to train an ASL model and evaluate its performance
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from backend.data_loader import ASLDataLoader
from backend.model import ASLModel

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix to visualize model performance"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved to 'confusion_matrix.png'")

def save_class_map(classes, output_path='models/class_map.json'):
    """Save class mapping to JSON file"""
    import json
    # Create class mapping (index -> letter)
    class_map = {i: classes[i] for i in range(len(classes))}
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save mapping
    with open(output_path, 'w') as f:
        json.dump(class_map, f)
    
    print(f"Class mapping saved to '{output_path}'")

def visualize_samples(data_loader, num_samples=5):
    """Visualize random samples from each class for verification"""
    classes = data_loader.classes
    plt.figure(figsize=(15, len(classes)*2))
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_loader.data_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Random sample
        samples = np.random.choice(images, min(num_samples, len(images)), replace=False)
        
        for j, img_file in enumerate(samples):
            if j >= num_samples:
                break
                
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(classes), num_samples, i*num_samples + j + 1)
            plt.imshow(img)
            plt.axis('off')
            if j == 0:
                plt.title(f'Class: {class_name}')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    print(f"Dataset samples visualization saved to 'dataset_samples.png'")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate ASL sign language detection model")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the ASL dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save the trained model")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Size to resize images to (square)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize dataset samples before training")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    print("Loading data from", args.data_dir)
    data_loader = ASLDataLoader(args.data_dir, img_size=(args.img_size, args.img_size))
    
    # Visualize dataset samples if requested
    if args.visualize:
        visualize_samples(data_loader)
    
    # Load and split the dataset
    X_train, X_test, y_train, y_test, classes = data_loader.load_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")
    print(f"Classes ({len(classes)}): {classes}")
    
    # Save the class mapping
    save_class_map(classes)
    
    # Create model
    print("Creating model...")
    model = ASLModel(input_shape=(args.img_size, args.img_size, 3), 
                    num_classes=len(classes))
    
    # Set up callbacks for better training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
    ]
    
    # Train model
    print("Training model...")
    history = model.train(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size, epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save model
    model_path = os.path.join(args.model_dir, 'asl_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    print("\nEvaluating model:")
    y_pred = model.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, classes)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plotted and saved as 'training_history.png'")
    
    # Final message
    print("\nTraining and evaluation complete!")
    print(f"Final validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model saved to {model_path}")
    print(f"Use 'python backend/api.py --model_path {model_path}' to start the API server.")

if __name__ == "__main__":
    main()
