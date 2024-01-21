import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class AttentionHead(Layer):
    def __init__(self, head_dim: int) -> None:
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
    
    def _scaled_dot_product_attention(self, q, k, v) -> tf.Tensor:
        """
        Computes scaled dot product attention.

        Args:
            q: Query tensor of shape [batch_size, seq_len_q, depth_q].
            k: Key tensor of shape [batch_size, seq_len_k, depth_k].
            v: Value tensor of shape [batch_size, seq_len_v, depth_v].

        Returns:
            Tensor of shape [batch_size, seq_len_q, depth_v] representing the output of scaled dot product attention.
        """
        s = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(k.shape[-1], dtype=tf.float32))
        w = tf.nn.softmax(s, axis=-1)
        return tf.matmul(w, v)
        
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
    def __init__(self, head_dim: int, num_heads: int) -> None:
        """
        Initializes the class instance.

        Parameters:
        - head_dim (int): The dimension of the head.
        - num_heads (int): The number of heads.

        Returns:
        - None
        """
        super().__init__()
        self.heads = [AttentionHead(head_dim) for _ in range(num_heads)]
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