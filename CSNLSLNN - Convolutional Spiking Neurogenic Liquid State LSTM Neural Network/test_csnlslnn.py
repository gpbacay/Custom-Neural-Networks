import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

class SpikingLNNLayer(tf.keras.layers.Layer):
    def __init__(self, reservoir_weights, input_weights, leak_rate, spike_threshold=1.0, **kwargs):
        super(SpikingLNNLayer, self).__init__(**kwargs)
        self.reservoir_weights = tf.Variable(reservoir_weights, dtype=tf.float32, trainable=False)
        self.input_weights = tf.Variable(input_weights, dtype=tf.float32, trainable=False)
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold

    def call(self, x):
        batch_size = tf.shape(x)[0]
        reservoir_dim = self.reservoir_weights.shape[0]
        state = tf.zeros((batch_size, reservoir_dim), dtype=tf.float32)
        
        input_part = tf.matmul(x, self.input_weights, transpose_b=True)
        reservoir_part = tf.matmul(state, self.reservoir_weights, transpose_b=True)
        state = (1 - self.leak_rate) * state + self.leak_rate * tf.tanh(input_part + reservoir_part)
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)
        return state

    def get_config(self):
        config = super(SpikingLNNLayer, self).get_config()
        config.update({
            "reservoir_weights": self.reservoir_weights.numpy().tolist(),
            "input_weights": self.input_weights.numpy().tolist(),
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
        })
        return config

    @classmethod
    def from_config(cls, config):
        reservoir_weights = np.array(config.pop("reservoir_weights"))
        input_weights = np.array(config.pop("input_weights"))
        return cls(reservoir_weights, input_weights, **config)

def test_model_with_image(model_path, img_path):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'SpikingLNNLayer': SpikingLNNLayer}
        )
        print("Model loaded successfully.")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"Image shape: {img_array.shape}")
        print("Image loaded and preprocessed successfully.")
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return
    
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Predictions: {predictions}")
        print(f"Predicted class: {predicted_class}")
    except Exception as e:
        print(f"Error predicting class: {e}")
        return
    
    try:
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {predicted_class}')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")

if __name__ == "__main__":
    model_path = 'TrainedModels/csnlslnn_mnist.keras'
    img_path = 'img_1.jpg'
    test_model_with_image(model_path, img_path)



# Run the script from the command line as follows:
# python test_csnlslnn.py
