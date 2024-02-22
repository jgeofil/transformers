import tensorflow as tf

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
        x = self.gamma * x + self.beta
        return x