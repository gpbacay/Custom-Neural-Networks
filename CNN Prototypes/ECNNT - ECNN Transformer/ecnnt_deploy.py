import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
try:
    MODEL_FILEPATH = 'Trained Models/ecnn_transformer_mnist.keras'
    model = load_model(MODEL_FILEPATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The {MODEL_FILEPATH} model file was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")
    exit()

def preprocess_image(image_path):
    try:
        # Load and preprocess an image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Image at path '{image_path}' could not be loaded.")
            return None
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=[0, -1])  # Shape to (1, 28, 28, 1)
        return image
    except Exception as e:
        print(f"An error occurred during image preprocessing: {str(e)}")
        return None

def predict(image_path):
    image = preprocess_image(image_path)
    if image is None:
        print("Skipping prediction due to preprocessing error.")
        return None
    
    try:
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)
        return predicted_digit
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    test_image_path = 'Test/img_5.png'
    predicted_digit = predict(test_image_path)
    if predicted_digit is not None:
        print(f'Predicted digit: {predicted_digit}')

# Efficient CNN Transformer (ECNN)
# python ecnnt_deploy.py
# Result: (Impressive)