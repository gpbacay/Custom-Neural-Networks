import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model

def efficientnet_block(inputs, filters, expansion_factor, stride):
    expanded_filters = filters * expansion_factor
    x = Conv2D(expanded_filters, kernel_size=1, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(expanded_filters, kernel_size=3, padding='same', strides=stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride == 1 and inputs.shape[-1] == x.shape[-1]:
        x = tf.keras.layers.Add()([inputs, x])
    return x

class SpatioTemporalRelationalLayer(tf.keras.layers.Layer):
    def __init__(self, num_relations, units, **kwargs):
        super().__init__(**kwargs)
        self.num_relations = num_relations
        self.units = units
        self.relation_networks = [Dense(units, activation='relu') for _ in range(num_relations)]
        self.temporal_network = Dense(units, activation='relu')

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.temporal_aggregation_weights = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='temporal_aggregation_weights'
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size, feature_dim = tf.shape(inputs)[0], inputs.shape[-1]
        relation_outputs = [network(inputs) for network in self.relation_networks]
        combined_relation_outputs = tf.stack(relation_outputs, axis=1)
        combined_relation_outputs = tf.reduce_sum(combined_relation_outputs, axis=1)
        temporal_aggregated = tf.tensordot(inputs, self.temporal_aggregation_weights, axes=[[-1], [0]])
        output = tf.add(combined_relation_outputs, temporal_aggregated)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_relations': self.num_relations,
            'units': self.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_cstart_model(input_shape, output_dim, num_relations=3, d_model=64, num_heads=4):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = efficientnet_block(x, 16, expansion_factor=1, stride=1)
    x = efficientnet_block(x, 24, expansion_factor=6, stride=2)
    x = efficientnet_block(x, 40, expansion_factor=6, stride=2)
    
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, x.shape[-1]))(x)
    
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    x = Flatten()(x)
    spatio_temporal_relational = SpatioTemporalRelationalLayer(num_relations, 128)(x)
    
    x = Dense(128, activation='relu')(spatio_temporal_relational)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
