import numpy as np
from PIL import Image
import tensorflow as tf
from lslnn_model import LNNStep  # Import your custom layer

# Load the trained model with custom objects
model = tf.keras.models.load_model(
    'TrainedModels/lslnn_model.keras',
    custom_objects={'LNNStep': LNNStep}
)

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize the image to 28x28 pixels (MNIST size)
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert image to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    
    # Reshape to (1, 28, 28) to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Path to the image file
image_path = 'img_1.jpg'
preprocessed_image = preprocess_image(image_path)

# Make prediction
predictions = model.predict(preprocessed_image)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=-1)
print(f"Predicted class: {predicted_class[0]}")


# Run the script from the command line as follows:
# python test1.py
