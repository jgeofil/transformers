import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout, LayerNormalization


class EmbeddingsLayer(Layer):
    def __init__(self, vocab_size: int, hidden_dim: int, position_dim: int=1000) -> None:
        """
        Initializes the class instance.

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - hidden_dim (int): The dimension of the hidden layer.

        Returns:
        - None
        """
        super().__init__()
        self.token_embedding = Embedding(vocab_size, hidden_dim)
        self.position_embedding = Embedding(position_dim, hidden_dim)
        self.layer_norm = LayerNormalization(epsilon=1e-12)
        self.dropout = Dropout(0.1)
    
    def call(self, inputs) -> tf.Tensor:
        """
        Applies embeddings to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying embeddings.
        """
        max_seq_len = inputs.shape[1]
        position_ids = tf.range(start=0, limit=max_seq_len, delta=1)
        
        position_emb = self.position_embedding(position_ids)
        token_emb = self.token_embedding(inputs)
        
        emb = self.layer_norm(token_emb + position_emb)
        return self.dropout(emb)