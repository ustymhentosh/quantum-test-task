import numpy as np
import random
from interfaces import DigitClassificationInterface

class CNNModel(DigitClassificationInterface):
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented for CNNModel.")

    def predict(self, image: np.ndarray) -> int:
        # CNN expects exactly 28x28x1 tensor
        # In a real scenario, you would pass this to PyTorch/TensorFlow
        
        # Dummy prediction for architecture demonstration
        return random.randint(0, 9)


class RandomForestModel(DigitClassificationInterface):
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented for RandomForestModel.")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # RF expects a 1-d numpy array of length 784
        return image.flatten()

    def predict(self, image: np.ndarray) -> int:
        processed_image = self._preprocess(image)
        
        # In a real scenario, you would pass processed_image (784,) to sklearn
        # e.g., return self.model.predict([processed_image])[0]
        
        # Dummy prediction
        return random.randint(0, 9)


class RandomModel(DigitClassificationInterface):
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented for RandomModel.")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # Random model expects a 10x10 numpy array, the center crop.
        # Center of 28 is 14. 14 - 5 = 9, 14 + 5 = 19.
        crop = image[9:19, 9:19, 0] 
        return crop

    def predict(self, image: np.ndarray) -> int:
        processed_image = self._preprocess(image)
        
        # Dummy prediction based on the 10x10 crop
        return random.randint(0, 9)