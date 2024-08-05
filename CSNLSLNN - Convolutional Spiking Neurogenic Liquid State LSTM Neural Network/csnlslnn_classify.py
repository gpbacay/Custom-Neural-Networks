import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt  # Import Matplotlib for image display
from csnlslnn_model import SpikingLNNStep  # Import the custom layer from your model
import sys

def load_model(model_path):
    """
    Load the trained model with custom objects.
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={'SpikingLNNStep': SpikingLNNStep}  # Adjust if using a different custom layer class name
    )

def preprocess_image(image_path):
    """
    Load and preprocess the image for model prediction.
    """
    try:
        # Load, resize, and normalize the image
        img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
        img_array = np.array(img, dtype='float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Add the channel dimension for CNN input
        return img_array, img  # Return the original image for display
    except FileNotFoundError:
        print(f"Error: The file {image_path} does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while processing the image: {e}")
        sys.exit(1)

def classify_image(model, image_path):
    """
    Classify the image using the provided model.
    """
    preprocessed_image, original_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    
    return predicted_class, original_image

def main():
    """
    Main function to load model, classify image, and handle exceptions.
    """
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = 'TrainedModels/csnlslnn_model.keras'  # Adjust to your model path

    try:
        # Load the trained model
        model = load_model(model_path)
        
        # Classify the image
        predicted_class, original_image = classify_image(model, image_path)
        
        print(f"The image is classified as: {predicted_class}")

        # Show the image
        plt.imshow(original_image, cmap='gray')
        plt.title(f"Predicted Class: {predicted_class}")
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Run the script from the command line as follows:
# python csnlslnn_classify.py img_5.png
