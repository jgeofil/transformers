import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

class FeedForward(Layer):
    def __init__(self, hidden_dim: int) -> None:
        """
        Initializes the class instance.

        Parameters:
        - hidden_dim (int): The dimension of the hidden layer.

        Returns:
        - None
        """
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation='gelu')
        self.dense2 = Dense(hidden_dim)
        self.dropout = Dropout(0.1)
    
    def call(self, inputs) -> tf.Tensor:
        """
        Applies feed forward network to the inputs.

        Args:
            inputs: The input tensor.

        Returns:
            The output tensor after applying feed forward network.
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dropout(x)