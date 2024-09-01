# model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

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

class GatedSLNNStep(tf.keras.layers.Layer):
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

            # Add new connections or prune existing ones here
            # Example: Dynamically resize the reservoir weights
            if self.reservoir_dim > self.max_reservoir_dim:
                self.reservoir_weights.assign(tf.pad(self.reservoir_weights, [[0, self.reservoir_dim - self.max_reservoir_dim], [0, 0]]))
            # Add your pruning logic here if needed

        # Ensure the state size matches the maximum reservoir dimension
        padded_state = tf.pad(state, [[0, 0], [0, self.max_reservoir_dim - self.reservoir_dim]])

        return padded_state, [padded_state]

    @property
    def state_size(self):
        return (self.max_reservoir_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'reservoir_dim': self.reservoir_dim,
            'input_dim': self.input_dim,
            'leak_rate': self.leak_rate,
            'spike_threshold': self.spike_threshold,
            'max_reservoir_dim': self.max_reservoir_dim
        })
        return config

class MetaFeatureLayer(tf.keras.layers.Layer):
    def __init__(self, self_modeling_mechanism, **kwargs):
        super().__init__(**kwargs)
        self.self_modeling_mechanism = self_modeling_mechanism

    def call(self, inputs):
        meta_features = self.self_modeling_mechanism.get_meta_features()
        meta_features = tf.convert_to_tensor(meta_features, dtype=tf.float32)
        meta_features = tf.reshape(meta_features, (1, -1))
        return tf.tile(meta_features, [tf.shape(inputs)[0], 1])

    def get_config(self):
        config = super().get_config()
        return config

def create_self_modeling_gslnn_model(input_shape, initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim, leak_rate, spike_threshold, output_dim):
    inputs = Input(shape=input_shape)

    self_modeling_mechanism = SelfModelingMechanism(initial_reservoir_dim, max_reservoir_dim, min_reservoir_dim)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    lnn_layer = tf.keras.layers.RNN(
        GatedSLNNStep(initial_reservoir_dim, x.shape[-1], leak_rate, spike_threshold, max_reservoir_dim, self_modeling_mechanism),
        return_sequences=True
    )

    def apply_self_modeling_gslnn(x):
        lnn_output = lnn_layer(tf.expand_dims(x, axis=1))
        return tf.squeeze(lnn_output, axis=1)

    lnn_output = Lambda(apply_self_modeling_gslnn, output_shape=(max_reservoir_dim,))(x)

    meta_features = MetaFeatureLayer(self_modeling_mechanism)(x)

    combined = tf.keras.layers.Concatenate()([lnn_output, meta_features])

    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
