import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self):
        # Load model once during initialization
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))
    
    def predict(self, image_path):
        try:
            # Load and preprocess image
            test_image = image.load_img(image_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)/255.0
            
            # Make prediction
            prediction = self.model.predict(test_image)
            result = np.argmax(prediction, axis=1)
            
            # Return both class and confidence
            confidence = float(np.max(prediction))
            if result[0] == 1:
                return [{"image": "Edible", "confidence": confidence}]
            else:
                return [{"image": "Poisonous", "confidence": confidence}]
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")