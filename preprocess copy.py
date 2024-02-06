import glob
import string
import tensorflow as tf
from tensorflow.keras.layers import StringLookup


class SpecialTokens:
    PADDING = '[PAD]'
    UNKNOWN = '[UNK]'
    START = '[STA]'
    END = '[END]'
    
    Tokens = [PADDING, UNKNOWN, START, END]

dataset = tf.constant([
    'sweetpotato',
    'lemoncake',
    'salmontomato'
])

#dataset = tf.data.Dataset.from_tensor_slices(dataset)

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

def add_special_tokens(tensor):
    print(tensor.shape)
    special_tokens_shape = tensor.shape[:-1] + [1]
    print(special_tokens_shape)
    start_tokens = tf.constant(ids_from_chars(SpecialTokens.START))
    end_tokens = tf.constant(ids_from_chars(SpecialTokens.END))
    return tf.concat((start_tokens, tensor, end_tokens), axis=-1)


def prepare_lm_inputs_labels(text):
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    padded = add_special_tokens(ids_from_chars(chars)).to_tensor()
    x = padded[:, :-1]
    y = padded[:, 1:]
    return x, y

dataset = prepare_lm_inputs_labels(dataset)

print(dataset)