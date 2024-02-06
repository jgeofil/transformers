
import glob
import string
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import keras_nlp
from keras_nlp.layers import StartEndPacker, TokenAndPositionEmbedding, TransformerDecoder
import numpy as np
np.set_printoptions(linewidth=200)

MAX_SEQ_LEN = 31

EMBED_DIM = 256

NUM_HEADS = 2
FEED_FORWARD_DIM = 256

class SpecialTokens:
    PADDING = '[PAD]'
    UNKNOWN = '[UNK]'
    START = '[STA]'
    END = '[END]'
    
    Tokens = [PADDING, UNKNOWN, START, END]

input_files = glob.glob('data/domains/*.txt.gz')

print('Input files:', len(input_files))

dataset = tf.data.TextLineDataset(input_files, compression_type='GZIP')
dataset = dataset.shuffle(buffer_size=1000000)
dataset = dataset.filter(lambda x: tf.strings.length(x) > 1)
dataset = dataset.batch(1024)

VOCAB = string.ascii_lowercase + string.digits + '-'
VOCAB = SpecialTokens.Tokens + [c for c in VOCAB]

ids_from_chars = StringLookup(
    vocabulary=VOCAB,
    mask_token=SpecialTokens.PADDING,
    oov_token=SpecialTokens.UNKNOWN
)

chars_from_ids = StringLookup(
    vocabulary=VOCAB,
    invert=True,
    mask_token=SpecialTokens.PADDING,
    oov_token=SpecialTokens.UNKNOWN
)

start_token = ids_from_chars(tf.constant(SpecialTokens.START))
end_token = ids_from_chars(tf.constant(SpecialTokens.END))

print(start_token)

packer = StartEndPacker(
    MAX_SEQ_LEN,
    start_value=SpecialTokens.START,
    end_value=SpecialTokens.END,
    pad_value=SpecialTokens.PADDING,
    return_padding_mask=False
)


def prepare_lm_inputs_labels(text):
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    padded = packer(chars)
    tokens = ids_from_chars(padded)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y

text_ds = dataset.map(prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
    
    
class CharTransformer(tf.keras.Model):
    def __init__(self, vocab_size:int, num_layers: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(
            vocabulary_size=len(VOCAB),
            sequence_length=MAX_SEQ_LEN,
            embedding_dim=EMBED_DIM,
            mask_zero=True,
        )
        self.decoders = [TransformerDecoder(num_heads=NUM_HEADS, intermediate_dim=FEED_FORWARD_DIM)
                       for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size, activation='relu')
        
    def call(self, inputs):
        x = self.embeddings(inputs)
        for layer in self.decoders:
            x = layer(x)
        return self.dense(x)
      
model = CharTransformer(vocab_size=len(VOCAB), num_layers=2, embedding_dim=EMBED_DIM, num_heads=NUM_HEADS)


perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[perplexity])

model.fit(text_ds, epochs=10, verbose=1, steps_per_epoch=1000)
        