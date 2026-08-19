import numpy as np
from classifier import DigitClassifier

def main():
    # dummy 28x28x1 image
    dummy_image = np.random.rand(28, 28, 1)

    algorithms = ['cnn', 'rf', 'rand']

    for algo in algorithms:
        print(f"--- Initializing {algo.upper()} Classifier ---")
        classifier = DigitClassifier(algorithm=algo)
        
        # unified predict method
        prediction = classifier.predict(dummy_image)
        print(f"Prediction from {algo}: {prediction}")
        
        # training raises the exception
        try:
            classifier._model.train()
        except NotImplementedError as e:
            print(f"Exception successfully caught: {e}\n")

if __name__ == "__main__":
    main()