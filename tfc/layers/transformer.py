import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dropout, LayerNormalization, Dense, Softmax
from layers.attention import MultiHeadAttention
from layers.dense import FeedForward
from layers.embedding import EmbeddingsLayer

class TransformerEncoderLayer(Layer):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.multihead = MultiHeadAttention(head_dim, num_heads)
        self.feedforward = FeedForward(head_dim * num_heads)
    
    def call(self, inputs) -> tf.Tensor:
        x = self.layer_norm_1(inputs)
        x += self.multihead(x)
        x += self.feedforward(self.layer_norm_2(x))
        return x
    
class TransformerEncoder(Layer):
    def __init__(self, vocab_size:int, num_layers: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embeddings = EmbeddingsLayer(vocab_size, embedding_dim)
        self.layers = [TransformerEncoderLayer(embedding_dim // num_heads, num_heads) for _ in range(num_layers)]
    
    def call(self, inputs) -> tf.Tensor:

        x = self.embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class TransformerDecoderLayer(Layer):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.multihead_1 = MultiHeadAttention(head_dim, num_heads, causal=True)
        self.feedforward = FeedForward(head_dim * num_heads)
    
    def call(self, inputs) -> tf.Tensor:
        x = self.layer_norm_1(inputs)
        x += self.multihead_1(x)
        x += self.feedforward(self.layer_norm_2(x))
        return x
    
class TransformerDecoder(Layer):
    def __init__(self, vocab_size:int, num_layers: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embeddings = EmbeddingsLayer(vocab_size, embedding_dim)
        self.layers = [TransformerDecoderLayer(embedding_dim // num_heads, num_heads) for _ in range(num_layers)]
        self.dense = Dense(vocab_size, activation='relu')
        self.softmax = Softmax()
    
    def call(self, inputs) -> tf.Tensor:
        x = self.embeddings(inputs)
        for layer in self.layers:
            x = layer(x)
        return self.dense(x)
    
class GenerativeDecoder(tf.keras.Model):
    def __init__(self, vocab_size:int, num_layers: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.decoder = TransformerDecoder(vocab_size, num_layers, embedding_dim, num_heads)
        
    def call(self, inputs) -> tf.Tensor:
        return self.decoder(inputs)