import cv2
import numpy as np
from csmgslnn_model import create_self_modeling_gslnn_model
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Image not found or unable to read.")
    
    # Resize image to the required input shape (28x28)
    image = cv2.resize(image, (28, 28))
    
    # Normalize image
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to fit model input
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image

def display_image_and_prediction(image_path, predicted_digit):
    # Display the image along with its predicted digit
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Image at path '{image_path}' could not be loaded.")
        return
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'Predicted digit: {predicted_digit}')
    plt.axis('off')
    plt.show()
    
def main():
    # Create the model
    model = create_self_modeling_gslnn_model(
        input_shape=(28, 28, 1),
        initial_reservoir_dim=100,
        max_reservoir_dim=200,
        min_reservoir_dim=50,
        leak_rate=0.1,
        spike_threshold=0.5,
        output_dim=10
    )
    
    # Load the model weights
    model.load_weights('Trained Models/csmgslnn_mnist.keras')

    # Path to the image you want to classify
    image_path = 'Test/img_5.png'
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Predict using the model
    prediction = model.predict(preprocessed_image)
    
    # Print the prediction
    print(f'Prediction: {np.argmax(prediction)}')
    display_image_and_prediction(image_path, np.argmax(prediction))

if __name__ == "__main__":
    main()



# CSM-GSLNN - Convolutional Self-Modeling Gated Spiking Liquid Neural Network
# python csmgslnn_deploy.py
# Remarks: Passed (with jsut 0.9904 accuracy!)
