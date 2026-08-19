import numpy as np
from models import CNNModel, RandomForestModel, RandomModel

class DigitClassifier:
    """
    Main classifier that routes predictions to the selected algorithm.
    """
    def __init__(self, algorithm: str):
        self.algorithm = algorithm.lower()
        
        # Instantiate the correct model based on the requested algorithm
        if self.algorithm == 'cnn':
            self._model = CNNModel()
        elif self.algorithm == 'rf':
            self._model = RandomForestModel()
        elif self.algorithm == 'rand':
            self._model = RandomModel()
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Supported algorithms are: 'cnn', 'rf', 'rand'."
            )

    def predict(self, image: np.ndarray) -> int:
        """
        Takes a 28x28x1 image and routes it to the underlying model.
        """
        # Validate unified input structure
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
            
        if image.shape != (28, 28, 1):
            raise ValueError(f"Input image must have shape (28, 28, 1), got {image.shape}")

        return self._model.predict(image)