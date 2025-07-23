import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np

def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for ASL sign detection
    """
    # Use Keras functional API for more flexibility
    inputs = keras.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class ASLModel:
    def __init__(self, input_shape=(64, 64, 3), num_classes=26):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = create_cnn_model(input_shape, num_classes)
        
    def train(self, X_train, y_train, X_test, y_test, batch_size=32, epochs=50, callbacks=None):
        # Data augmentation for training
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        
        # Generate augmented training data
        datagen.fit(X_train)
        
        # Train the model with augmentation
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        # Ensure image has correct shape
        if len(image.shape) == 3:  # Add batch dimension if needed
            image = np.expand_dims(image, axis=0)
        return self.model.predict(image)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = keras.models.load_model(path)
