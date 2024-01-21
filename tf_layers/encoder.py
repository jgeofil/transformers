import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout, LayerNormalization
from tf_layers.attention import MultiHeadAttention
from tf_layers.dense import FeedForward
from tf_layers.embedding import EmbeddingsLayer

class TransformerEncoderLayer(Layer):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        """
        Initializes the class instance.

        Parameters:
        - head_dim (int): The dimension of the head.
        - num_heads (int): The number of heads.
        - hidden_dim (int): The dimension of the hidden layer.

        Returns:
        - None
        """
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.multihead = MultiHeadAttention(head_dim, num_heads)
        self.feedforward = FeedForward(head_dim * num_heads)
    
    def call(self, inputs) -> tf.Tensor:
        """
        Applies encoder to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying encoder.
        """
        x = self.layer_norm_1(inputs)
        x += self.multihead(x)
        x += self.feedforward(self.layer_norm_2(x))
        return x
    

    
class TransformerEncoder(Layer):
    def __init__(self, vocab_size:int, num_layers: int, embedding_dim: int, num_heads: int) -> None:
        """
        Initializes the class instance.

        Parameters:
        - num_layers (int): The number of layers.
        - embedding_dim (int): The dimension of the embeddings.
        - num_heads (int): The number of heads.

        Returns:
        - None
        """
        super().__init__()
        self.embeddings = EmbeddingsLayer(vocab_size, embedding_dim)
        self.layers = [TransformerEncoderLayer(embedding_dim // num_heads, num_heads) for _ in range(num_layers)]
    
    def call(self, inputs) -> tf.Tensor:
        """
        Applies encoder to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying encoder.
        """
        x = self.embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        return x