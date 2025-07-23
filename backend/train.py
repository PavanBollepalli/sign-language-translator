import os
import argparse
import matplotlib.pyplot as plt
from data_loader import ASLDataLoader
from model import ASLModel

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main(args):
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = ASLDataLoader(args.data_dir, img_size=(args.img_size, args.img_size))
    X_train, X_test, y_train, y_test, classes = data_loader.load_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")
    print(f"Classes: {classes}")
    
    # Create and train model
    print("Creating model...")
    model = ASLModel(input_shape=(args.img_size, args.img_size, 3), 
                     num_classes=len(classes))
    
    print("Training model...")
    history = model.train(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size, epochs=args.epochs
    )
    
    # Save model
    model_path = os.path.join(args.model_dir, 'asl_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plot_history(history)
    print("Training history plotted and saved as 'training_history.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL sign language detection model")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the ASL dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save the trained model")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Size to resize images to (square)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
