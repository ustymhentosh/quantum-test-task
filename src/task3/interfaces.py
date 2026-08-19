import abc
import numpy as np

class DigitClassificationInterface(abc.ABC):
    """
    Interface for all digit classification models.
    Any new model must implement the predict and train methods.
    """
    
    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit from a 28x28x1 image.
        
        Args:
            image (np.ndarray): The input image of shape (28, 28, 1).
            
        Returns:
            int: The predicted digit (0-9).
        """
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        Trains the model. 
        Raises NotImplementedError as per requirements.
        """
        pass