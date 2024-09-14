import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, GlobalAveragePooling2D, RNN, Layer
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class AutonomousReservoirCell(Layer):
    def __init__(self, initial_size, max_size, spectral_radius, leak_rate, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.initial_size = initial_size
        self.max_size = max_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_dim = input_dim
        self.current_size = initial_size
        self.state_size = self.max_size
        self.output_size = self.max_size

    def build(self, input_shape):
        self.reservoir_weights = self.add_weight(
            shape=(self.max_size, self.max_size),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=False,
            name='reservoir_weights'
        )
        self.input_weights = self.add_weight(
            shape=(self.max_size, self.input_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=False,
            name='input_weights'
        )
        self._adjust_spectral_radius()
        self.built = True

    def _adjust_spectral_radius(self):
        eigenvalues = tf.linalg.eigvals(self.reservoir_weights[:self.current_size, :self.current_size])
        spectral_radius = tf.reduce_max(tf.abs(eigenvalues))
        scaling_factor = self.spectral_radius / spectral_radius
        self.reservoir_weights.assign(self.reservoir_weights * scaling_factor)

    def call(self, inputs, states):
        prev_state = states[0][:, :self.current_size]
        input_contrib = tf.matmul(inputs, tf.transpose(self.input_weights[:self.current_size]))
        reservoir_contrib = tf.matmul(prev_state, self.reservoir_weights[:self.current_size, :self.current_size])
        
        new_state = (1 - self.leak_rate) * prev_state + self.leak_rate * tf.tanh(input_contrib + reservoir_contrib)
        
        activity = tf.reduce_mean(tf.abs(new_state), axis=0)
        
        # Use tf.cond for symbolic control flow
        new_state = tf.cond(
            tf.reduce_min(activity) < 0.01,
            lambda: self._expand_reservoir() or new_state,
            lambda: new_state
        )
        
        new_state = tf.cond(
            tf.reduce_max(activity) > 0.99,
            lambda: self._prune_reservoir() or new_state,
            lambda: new_state
        )
        
        padded_state = tf.pad(new_state, [[0, 0], [0, self.max_size - self.current_size]])
        return padded_state, [padded_state]

    def _expand_reservoir(self):
        new_neurons = min(int(self.current_size * 0.1), self.max_size - self.current_size)
        if new_neurons > 0:
            self.current_size += new_neurons
            self._adjust_spectral_radius()

    def _prune_reservoir(self):
        prune_neurons = min(int(self.current_size * 0.1), self.current_size - self.initial_size)
        if prune_neurons > 0:
            self.current_size -= prune_neurons
            self._adjust_spectral_radius()

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_size': self.initial_size,
            'max_size': self.max_size,
            'spectral_radius': self.spectral_radius,
            'leak_rate': self.leak_rate,
            'input_dim': self.input_dim,
        })
        return config

class AutonomousSelfModelingCallback(Callback):
    def __init__(self, reservoir_cell):
        super().__init__()
        self.reservoir_cell = reservoir_cell
        self.performance_history = []

    def on_epoch_end(self, epoch, logs=None):
        self.performance_history.append(logs.get('val_classification_output_accuracy', 0))
        
        if len(self.performance_history) > 5:
            recent_performance = self.performance_history[-5:]
            if all(x >= y for x, y in zip(recent_performance, recent_performance[1:])):
                self.reservoir_cell._expand_reservoir()
            elif all(x <= y for x, y in zip(recent_performance, recent_performance[1:])):
                self.reservoir_cell._prune_reservoir()

class AutonomousFeedbackModulationLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_dense = Dense(units, activation='tanh')
        self.feedback_dense = Dense(units, activation='sigmoid')
        self.output_dense = Dense(units)

    def call(self, inputs):
        state, self_model_prediction = inputs
        state = self.state_dense(state)
        feedback = self.feedback_dense(self_model_prediction)
        modulated_state = state * feedback
        return self.output_dense(modulated_state)

class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config

def create_autonomous_self_modeling_nn(input_shape, output_dim, initial_reservoir_size=512, max_reservoir_size=4096):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = GlobalAveragePooling2D()(x)
    
    x = ExpandDimsLayer(axis=1)(x)
    
    reservoir_cell = AutonomousReservoirCell(
        initial_size=initial_reservoir_size,
        max_size=max_reservoir_size,
        spectral_radius=0.9,
        leak_rate=0.1,
        input_dim=x.shape[-1]
    )
    x = RNN(reservoir_cell, return_sequences=False)(x)
    
    self_model_output = Dense(np.prod(input_shape), name="self_model_output")(x)
    
    x = AutonomousFeedbackModulationLayer(units=256)([x, self_model_output])
    
    classification_output = Dense(output_dim, activation='softmax', name="classification_output")(x)
    
    model = Model(inputs, [classification_output, self_model_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'categorical_crossentropy',
            'self_model_output': 'mse'
        },
        loss_weights={
            'classification_output': 1.0,
            'self_model_output': 0.1
        },
        metrics={
            'classification_output': 'accuracy'
        }
    )
    
    return model, reservoir_cell

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_val = x_val.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    model, reservoir_cell = create_autonomous_self_modeling_nn(input_shape=(28, 28, 1), output_dim=10)
    
    early_stopping = EarlyStopping(monitor='val_classification_output_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_classification_output_accuracy', factor=0.2, patience=5)
    self_modeling_callback = AutonomousSelfModelingCallback(reservoir_cell)
    
    history = model.fit(
        x_train,
        {'classification_output': y_train, 'self_model_output': x_train.reshape((x_train.shape[0], -1))},
        validation_data=(x_val, {'classification_output': y_val, 'self_model_output': x_val.reshape((x_val.shape[0], -1))}),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr, self_modeling_callback]
    )
    
    test_loss, test_acc = model.evaluate(x_test, {'classification_output': y_test, 'self_model_output': x_test.reshape((x_test.shape[0], -1))})
    print(f"Test accuracy: {test_acc:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['classification_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_classification_output_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()

if __name__ == '__main__':
    main()


# Autonomous Self-Modeling Neural Network (ASMNN)
# python asmnn_mnist.py
# Test Accuracy: "slow" 14 mins per epoch