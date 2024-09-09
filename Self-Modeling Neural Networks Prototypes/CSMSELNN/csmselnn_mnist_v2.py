import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom Layers
class SpikingElasticLNNStep(tf.keras.layers.Layer):
    def __init__(self, initial_reservoir_size, input_dim, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, **kwargs):
        super(SpikingElasticLNNStep, self).__init__(**kwargs)
        self.initial_reservoir_size = initial_reservoir_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        
        # Define the state size and output size
        self.state_size = initial_reservoir_size
        self.output_size = initial_reservoir_size

    def build(self, input_shape):
        self.reservoir_size = self.initial_reservoir_size
        self.W = self.add_weight(shape=(self.input_dim, self.reservoir_size), initializer='glorot_uniform', trainable=False)
        self.W_reservoir = self.add_weight(shape=(self.reservoir_size, self.reservoir_size), initializer='glorot_uniform', trainable=False)
        self.b = self.add_weight(shape=(self.reservoir_size,), initializer='zeros', trainable=False)
        super(SpikingElasticLNNStep, self).build(input_shape)

    def call(self, inputs, states):
        reservoir_state = states[0]
        new_reservoir_state = self.leak_rate * reservoir_state + (1 - self.leak_rate) * tf.nn.tanh(tf.matmul(inputs, self.W) + tf.matmul(reservoir_state, self.W_reservoir) + self.b)
        spikes = tf.cast(new_reservoir_state > self.spike_threshold, tf.float32)
        new_reservoir_state = tf.where(spikes > 0, 0.0, new_reservoir_state)
        return new_reservoir_state, [new_reservoir_state]

    def get_config(self):
        config = super(SpikingElasticLNNStep, self).get_config()
        config.update({
            'initial_reservoir_size': self.initial_reservoir_size,
            'input_dim': self.input_dim,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_reservoir_dim': self.max_reservoir_dim
        })
        return config

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# Custom Callback
class SelfModelingCallback(Callback):
    def __init__(self, selnn_step_layer, performance_metric='val_classification_output_accuracy'):
        super(SelfModelingCallback, self).__init__()
        self.selnn_step_layer = selnn_step_layer
        self.performance_metric = performance_metric

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve and log state of the custom layer
        state = self.selnn_step_layer.W.numpy()
        print(f" - Epoch {epoch+1}: Reservoir state mean: {np.mean(state):.4f}, Reservoir state std: {np.std(state):.4f}")

        if self.performance_metric in logs:
            metric_value = logs[self.performance_metric]
            print(f"Performance Metric ({self.performance_metric}): {metric_value:.4f}")

def create_csmselnn_model(input_shape, initial_reservoir_size, spectral_radius, leak_rate, spike_threshold, max_reservoir_dim, output_dim):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    x = Flatten()(x)
    
    # Create custom layer with a specific name
    selnn_step_layer = SpikingElasticLNNStep(
        initial_reservoir_size=initial_reservoir_size,
        input_dim=x.shape[-1],
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        spike_threshold=spike_threshold,
        max_reservoir_dim=max_reservoir_dim,
        name='spiking_elastic_lnns_step'
    )
    
    expanded_inputs = ExpandDimsLayer(axis=1)(x)
    rnn_layer = RNN(selnn_step_layer, return_sequences=False)
    selnn_output = rnn_layer(expanded_inputs)
    selnn_output = Flatten()(selnn_output)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(selnn_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    
    predicted_hidden = Dense(np.prod(input_shape), name="self_modeling_output")(x)
    
    outputs = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = Model(inputs, [outputs, predicted_hidden])
    
    return model

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype(np.float32) / 255.0
    x_test = np.expand_dims(x_test, -1).astype(np.float32) / 255.0
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    input_shape = x_train.shape[1:]
    output_dim = 10  # Number of classes
    
    model = create_csmselnn_model(
        input_shape=input_shape,
        initial_reservoir_size=512,
        max_reservoir_dim=4096,
        spectral_radius=0.5,
        leak_rate=0.1,
        spike_threshold=0.5,
        output_dim=output_dim
    )
    
    model.compile(
        optimizer='adam',
        loss={'classification_output': 'sparse_categorical_crossentropy', 'self_modeling_output': 'mean_squared_error'},
        metrics={'classification_output': 'accuracy'}
    )
    
    # Reference the RNN layer that wraps the SpikingElasticLNNStep
    rnn_layer = model.get_layer('rnn')
    selnn_step_layer = rnn_layer.cell  # Access the SpikingElasticLNNStep through the RNN cell
    
    self_modeling_callback = SelfModelingCallback(
        selnn_step_layer=selnn_step_layer,
        performance_metric='val_classification_output_accuracy'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_classification_output_accuracy', 
        patience=5, 
        restore_best_weights=True, 
        mode='max'  # Explicitly set mode to 'max'
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.5, patience=3, mode='max')
    
    history = model.fit(
        x_train, 
        {'classification_output': y_train, 'self_modeling_output': x_train.reshape(x_train.shape[0], -1)},
        epochs=10,
        batch_size=64,
        validation_data=(x_val, {'classification_output': y_val, 'self_modeling_output': x_val.reshape(x_val.shape[0], -1)}),
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_modeling_output': x_test.reshape(x_test.shape[0], -1)})
    print(f"Test accuracy: {test_acc:.4f}")

    plt.plot(history.history['classification_output_accuracy'])
    plt.plot(history.history['val_classification_output_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()


# Convolutional Self-Modeling Spiking Elastic Liquid Neural Network (CSMSELNN) version 2
# python csmselnn_mnist_v2.py
# Test Accuracy: 0.9920