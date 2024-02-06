import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np
class AttentionHead(Layer):
    def __init__(self, head_dim: int, causal=False) -> None:
        """
        Initializes the class instance.

        Parameters:
        - head_dim (int): The dimension of the head.

        Returns:
        - None
        """
        super().__init__()
        self.q = Dense(head_dim)
        self.k = Dense(head_dim)
        self.v = Dense(head_dim)
        
        self.causal = causal
    
    def _lower_triangular_mask(self, shape):

        row_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-2)
        col_index = tf.cumsum(tf.ones(shape=shape, dtype=tf.int32), axis=-1)
        return tf.greater_equal(row_index, col_index)
    
    def _scaled_dot_product_attention(self, query, key, value) -> tf.Tensor:
        """
        Computes scaled dot product attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len_q, depth_q].
            key: Key tensor of shape [batch_size, seq_len_k, depth_k].
            value: Value tensor of shape [batch_size, seq_len_v, depth_v].
            mask: Mask tensor of shape [batch_size, seq_len_q, seq_len_k].

        Returns:
            Tensor of shape [batch_size, seq_len_q, depth_v] representing the output of scaled dot product attention.
        """
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(key.shape[-1], dtype=tf.float32))
        if self.causal:
            scores_shape = tf.shape(scores)
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0
            )
            causal_mask = self._lower_triangular_mask(causal_mask_shape)
            padding_mask = tf.logical_not(causal_mask)
            if scores.dtype is tf.float16:
                scores -= 65504.0 * tf.cast(padding_mask, dtype=scores.dtype)
            else:
                scores -= 1.0e9 * tf.cast(padding_mask, dtype=scores.dtype)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, value)
        
    def call(self, inputs) -> tf.Tensor:
        """
        Applies scaled dot product attention to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying scaled dot product attention.
        """
        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)
        
        return self._scaled_dot_product_attention(q, k, v)

class MultiHeadAttention(Layer):
    def __init__(self, head_dim: int, num_heads: int, causal=False) -> None:
        """
        Initializes the class instance.

        Parameters:
        - head_dim (int): The dimension of the head.
        - num_heads (int): The number of heads.

        Returns:
        - None
        """
        super().__init__()
        self.heads = [AttentionHead(head_dim, causal=causal) for _ in range(num_heads)]
        self.dense = Dense(head_dim * num_heads)
    
    def call(self, inputs) -> tf.Tensor:
        """
        Applies multi-head attention to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying multi-head attention.
        """
        concat = tf.concat([head(inputs) for head in self.heads], axis=-1)
        return self.dense(concat)