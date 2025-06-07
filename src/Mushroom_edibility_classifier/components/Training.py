from Mushroom_edibility_classifier.entity.config_entity import TrainingConfig
from pathlib import Path
import os
import urllib.request as request
from zipfile import ZipFile
import time
import tensorflow as tf
from typing import List, Optional



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        tf.config.run_functions_eagerly(True) 
    
    def get_base_model(self):
        """Load the base model for training"""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def create_data_generators(self):
        """Create data generators for pre-split train/val/test datasets"""
        # Common preprocessing parameters
        datagen_kwargs = dict(
            rescale=1./255,
        )

        # Common flow parameters
        flow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'
        )

        # Validation generator (no augmentation)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
        self.val_generator = val_datagen.flow_from_directory(
            directory=self.config.validation_data,  # Path to validation folder
            shuffle=False,
            **flow_kwargs
        )

        # Training generator (with optional augmentation)
        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagen_kwargs
            )
        else:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

        self.train_generator = train_datagen.flow_from_directory(
            directory=self.config.training_data,  # Path to train folder
            shuffle=True,
            **flow_kwargs
        )

        # Optional test generator
        if hasattr(self.config, 'test_data'):
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
            self.test_generator = test_datagen.flow_from_directory(
                directory=self.config.test_data,
                shuffle=False,
                **flow_kwargs
            )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model"""
        model.save(path)

    def train(self, callback_list: Optional[List[tf.keras.callbacks.Callback]] = None):
        """Train the model with callbacks"""
        if callback_list is None:
            callback_list = []

        # Calculate steps
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.val_generator.samples // self.val_generator.batch_size
        self.model.optimizer = tf.keras.optimizers.Adam()

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=callback_list
        )

        # Save and return training history
        self.save_model(self.config.trained_model_path, self.model)
        
