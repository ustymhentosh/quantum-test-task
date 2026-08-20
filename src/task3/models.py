import random

import numpy as np
import torch
from interfaces import DigitClassificationInterface
from sklearn.ensemble import RandomForestClassifier
from torch import nn


class CNNModel(DigitClassificationInterface):
    def __init__(self):
        # Untrained PyTorch CNN structure
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 28 * 28, 10)
        )
        self.model.eval()

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented for CNNModel.")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected input shape (28, 28, 1), got {image.shape}")

        # Convert (28, 28, 1) -> PyTorch batch layout (1, 1, 28, 28)
        tensor_input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            logits = self.model(tensor_input)
            predicted_class = torch.argmax(logits, dim=1).item()

        return int(predicted_class)


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=5, random_state=42)
        dummy_x = np.random.rand(10, 784)
        dummy_y = np.random.randint(0, 10, size=10)
        self.model.fit(dummy_x, dummy_y)
    
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training is not implemented for RandomForestModel.")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # RF expects a 1-d numpy array of length 784
        return image.flatten()

    def predict(self, image: np.ndarray) -> int:
        processed_image = self._preprocess(image)
        prediction = self.model.predict([processed_image])[0]
        return int(prediction)


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
