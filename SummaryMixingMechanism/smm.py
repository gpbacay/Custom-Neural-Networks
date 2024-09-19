import tensorflow as tf

class SummaryMixing(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=None, dropout_rate=0.1):
        super(SummaryMixing, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Layers for local transformation
        self.local_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.local_dense2 = tf.keras.layers.Dense(d_model)
        self.local_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layers for summary function
        self.summary_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.summary_dense2 = tf.keras.layers.Dense(d_model)
        self.summary_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layers for combiner function
        self.combiner_dense1 = tf.keras.layers.Dense(self.d_ff, activation='gelu')
        self.combiner_dense2 = tf.keras.layers.Dense(d_model)
        self.combiner_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Local transformation
        local_output = self.local_dense1(inputs)
        local_output = self.local_dense2(local_output)
        local_output = self.local_dropout(local_output, training=training)
        
        # Summary function
        summary = self.summary_dense1(inputs)
        summary = self.summary_dense2(summary)
        summary = self.summary_dropout(summary, training=training)
        
        # Calculate mean summary
        mean_summary = tf.reduce_mean(summary, axis=1, keepdims=True)
        
        # Repeat mean summary for each time step
        mean_summary = tf.tile(mean_summary, [1, tf.shape(inputs)[1], 1])
        
        # Combine local and summary information
        combined = tf.concat([local_output, mean_summary], axis=-1)
        output = self.combiner_dense1(combined)
        output = self.combiner_dense2(output)
        output = self.combiner_dropout(output, training=training)
        
        # Residual connection and layer normalization
        return self.layer_norm(inputs + output)

# Example usage
input_shape = (batch_size, sequence_length, input_dim)
d_model = 512

inputs = tf.keras.Input(shape=input_shape[1:])
summary_mixing_layer = SummaryMixing(d_model)
outputs = summary_mixing_layer(inputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)