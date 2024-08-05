from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image

class DataEmbedder:
    """
    A class to embed MNIST images using a pre-trained ResNet50 model.
    """

    def __init__(self):
        """
        Initialize the DataEmbedder with a ResNet50 model.
        """
        self.image_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def embed_mnist(self, mnist_image):
        """
        Convert a single MNIST image to an embedding using ResNet50.

        Parameters:
        - mnist_image (numpy.ndarray): A 28x28 grayscale image from MNIST dataset.

        Returns:
        - numpy.ndarray: The ResNet50 embedding of the input image.
        """
        if mnist_image.shape != (28, 28):
            raise ValueError("Input image must be of shape (28, 28).")
        
        # Convert MNIST image to 3 channels and resize
        mnist_image = np.stack([mnist_image] * 3, axis=-1)  # Convert to RGB
        mnist_image = Image.fromarray(mnist_image.astype('uint8')).resize((224, 224))
        
        # Prepare the image for the model
        img_array = keras_image.img_to_array(mnist_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Generate embeddings
        embeddings = self.image_model.predict(img_array)
        return embeddings

# Example usage
if __name__ == "__main__":
    embedder = DataEmbedder()

    # Load MNIST data
    (mnist_train_images, _), (mnist_test_images, _) = mnist.load_data()

    # Embed a sample MNIST image
    mnist_image = mnist_test_images[0]
    mnist_embedding = embedder.embed_mnist(mnist_image)
    print("MNIST Embedding:", mnist_embedding)
