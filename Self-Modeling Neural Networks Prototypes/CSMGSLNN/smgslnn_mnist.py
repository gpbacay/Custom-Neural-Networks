import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SelfModelingMechanism:
    def __init__(self, initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim):
        self.reservoir_dim = initial_reservoir_dim
        self.max_reservoir_dim = max_reservoir_dim
        self.min_reservoir_dim = min_reservoir_dim
        self.performance_history = []
        self.structure_history = []
        self.num_meta_features = 4

    def adapt(self, state, performance):
        self.performance_history.append(performance)
        self.structure_history.append(self.reservoir_dim)

        if len(self.performance_history) >= 5:
            recent_trend = np.mean(self.performance_history[-5:]) - np.mean(self.performance_history[-10:-5])

            if recent_trend > 0.01:
                self.reservoir_dim = min(int(self.reservoir_dim * 1.1), self.max_reservoir_dim)
            elif recent_trend < -0.01:
                self.reservoir_dim = max(int(self.reservoir_dim * 0.9), self.min_reservoir_dim)

        new_state = tf.image.resize(tf.expand_dims(state, -1), [tf.shape(state)[0], self.reservoir_dim])
        return tf.squeeze(new_state, -1)

    def get_meta_features(self):
        if len(self.performance_history) < 10:
            return [0.0, 0.0, self.reservoir_dim / self.max_reservoir_dim, 0.0]
        return [
            np.mean(self.performance_history[-10:]),
            np.std(self.performance_history[-10:]),
            self.reservoir_dim / self.max_reservoir_dim,
            np.mean(self.structure_history[-10:]) / self.max_reservoir_dim
        ]

class AdaptiveGatedSLNNStep(Layer):
    def __init__(self, reservoir_dim, input_dim, leak_rate, spike_threshold, max_reservoir_dim, self_modeling_mechanism, **kwargs):
        super().__init__(**kwargs)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.leak_rate = leak_rate
        self.spike_threshold = spike_threshold
        self.max_reservoir_dim = max_reservoir_dim
        self.self_modeling_mechanism = self_modeling_mechanism

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.max_reservoir_dim, self.max_reservoir_dim),
            initializer='glorot_uniform',
            name='reservoir_weights',
            trainable=True
        )
        self.input_weights = self.add_weight(
            shape=(self.max_reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='input_weights',
            trainable=True
        )
        self.gate_weights = self.add_weight(
            shape=(3 * self.max_reservoir_dim, self.input_dim),
            initializer='glorot_uniform',
            name='gate_weights',
            trainable=True
        )

    def call(self, inputs, states, training=None):
        prev_state = states[0][:, :self.reservoir_dim]

        # Compute input, reservoir, and gate parts of the state update
        input_part = tf.matmul(inputs, self.input_weights[:self.reservoir_dim, :], transpose_b=True)
        reservoir_part = tf.matmul(prev_state, self.reservoir_weights[:self.reservoir_dim, :self.reservoir_dim], transpose_b=True)
        gate_part = tf.matmul(inputs, self.gate_weights[:3 * self.reservoir_dim, :], transpose_b=True)

        # Split gate activations into input, forget, and output gates
        i_gate, f_gate, o_gate = tf.split(tf.sigmoid(gate_part), 3, axis=-1)

        # Update state with gating and reservoir dynamics
        state = (1 - self.leak_rate) * (f_gate * prev_state) + self.leak_rate * tf.tanh(i_gate * (input_part + reservoir_part))
        state = o_gate * state

        # Apply threshold to produce discrete spikes
        spikes = tf.cast(tf.greater(state, self.spike_threshold), dtype=tf.float32)
        state = tf.where(spikes > 0, state - self.spike_threshold, state)

        # Adapt the reservoir size and connections dynamically
        if training:
            performance = tf.reduce_mean(tf.abs(state))  # Simple performance metric
            state = self.self_modeling_mechanism.adapt(state, performance)
            self.reservoir_dim = self.self_modeling_mechanism.reservoir_dim

        # Ensure the state size matches the maximum reservoir dimension
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - self.reservoir_dim]])

        return padded_state, [padded_state]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "leak_rate": self.leak_rate,
            "spike_threshold": self.spike_threshold,
            "max_dynamic_reservoir_dim": self.max_dynamic_reservoir_dim,
            "spatiotemporal_reservoir_weights": self.spatiotemporal_reservoir_weights.tolist(),
            "spatiotemporal_input_weights": self.spatiotemporal_input_weights.tolist(),
            "spiking_gate_weights": self.spiking_gate_weights.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        spatiotemporal_reservoir_weights = np.array(config.pop('spatiotemporal_reservoir_weights'))
        spatiotemporal_input_weights = np.array(config.pop('spatiotemporal_input_weights'))
        spiking_gate_weights = np.array(config.pop('spiking_gate_weights'))
        return cls(spatiotemporal_reservoir_weights, spatiotemporal_input_weights, spiking_gate_weights, **config)

class MetaFeatureLayer(Layer):
    def __init__(self, self_modeling_mechanism, **kwargs):
        super().__init__(**kwargs)
        self.self_modeling_mechanism = self_modeling_mechanism

    def call(self, inputs):
        meta_features = self.self_modeling_mechanism.get_meta_features()
        meta_features = tf.convert_to_tensor(meta_features, dtype=tf.float32)
        meta_features = tf.reshape(meta_features, (1, -1))
        return tf.tile(meta_features, [tf.shape(inputs)[0], 1])

def create_self_modeling_gslnn_model(input_dim, initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim, leak_rate, spike_threshold, output_dim):
    inputs = Input(shape=(input_dim,))

    self_modeling_mechanism = SelfModelingMechanism(initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim)

    lnn_layer = tf.keras.layers.RNN(
        AdaptiveGatedSLNNStep(initial_reservoir_dim, input_dim, leak_rate, spike_threshold, max_reservoir_dim, self_modeling_mechanism),
        return_sequences=True
    )

    def apply_self_modeling_gslnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return Flatten()(lnn_output)

    lnn_output = Lambda(apply_self_modeling_gslnn)(inputs)

    meta_features = MetaFeatureLayer(self_modeling_mechanism)(inputs)

    combined = tf.keras.layers.Concatenate()([lnn_output, meta_features])

    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Data preprocessing function
def preprocess_data(x):
    x = x.astype(np.float32) / 255.0
    return x.reshape(-1, 28 * 28)

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = preprocess_data(x_train)
    x_val = preprocess_data(x_val)
    x_test = preprocess_data(x_test)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)

    input_dim = 28 * 28
    initial_reservoir_dim = 512
    max_reservoir_dim = 1024
    min_reservoir_dim = 256
    leak_rate = 0.3
    spike_threshold = 0.5
    output_dim = 10
    num_epochs = 10
    batch_size = 64

    model = create_self_modeling_gslnn_model(input_dim, initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim, leak_rate, spike_threshold, output_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    ]

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy:.4f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



# Self-Modeling Gated Spiking Liquid Neural Network (SM-GSLNN)
# without adding neurons/pruning connections mechanisms
# python smgslnn_mnist.py
# Test Accuracy: 0.9610